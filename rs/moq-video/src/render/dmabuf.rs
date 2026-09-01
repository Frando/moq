//! Zero-copy import of packed Linux DMA-BUFs into wgpu's Vulkan backend.
//!
//! wgpu owns the Vulkan image and imported fd once wrapping succeeds. The
//! [`DmaBuf`] itself rides with the submitted render work separately, keeping
//! the dequeued producer buffer out of PipeWire's pool until the GPU is done.
//!
//! wgpu-hal's importer takes one stride and one offset, so a buffer Vulkan will
//! not take as a single plane has no route through it: a modifier carrying an
//! auxiliary compression plane, or a driver that does not list the modifier at
//! all. With the `vaapi` feature such a buffer gets a second chance, blitted on
//! the GPU into an allocation the driver lays out itself (see
//! [`moq_vaapi::vpp`]). The direct import is always tried first, since it copies
//! nothing at all.

use wgpu::hal::MemoryFlags;

use super::source::{Layout, Source};
use crate::{DmaBuf, DrmFormat, Error, Size};

fn err(message: impl std::fmt::Display) -> Error {
	Error::Render(anyhow::anyhow!("{message}"))
}

/// The DMA-BUF import path's state between frames.
#[derive(Default)]
pub(super) struct Import {
	#[cfg(feature = "vaapi")]
	vpp: Vpp,
}

impl Import {
	/// Alias one DMA-BUF allocation as a sampled Vulkan texture.
	pub fn import(&mut self, device: &wgpu::Device, buffer: &DmaBuf) -> Result<Option<Source>, Error> {
		if !device
			.features()
			.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
		{
			return Ok(None);
		}

		// SAFETY: the guard is only asked whether it exists, and drops here.
		// A device carrying the feature above is Vulkan-backed, so this is a
		// belt-and-braces check rather than a route anything takes.
		if unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_none() {
			return Ok(None);
		}

		let format = texture_format(buffer.format())?;
		let size = Size::new(buffer.width(), buffer.height());

		// One export, and so one wait on the producer's write fence, however
		// many import attempts follow it.
		let export = buffer
			.export()
			.map_err(|e| Error::Render(anyhow::Error::new(e).context("export DMA-BUF")))?;
		let (fd, lease) = export.into_parts();
		#[cfg(feature = "vaapi")]
		let spare = fd
			.try_clone()
			.map_err(|e| Error::Render(anyhow::Error::new(e).context("duplicate DMA-BUF")))?;

		let direct = match buffer.planes() {
			// SAFETY: `fd` is a fresh duplicate of this live DMA-BUF. Export
			// waited for producer writes, and the format, modifier, extent,
			// stride, and offset come from the producer's buffer metadata.
			[plane] => unsafe {
				adopt(
					device,
					format,
					size,
					buffer.modifier(),
					plane.stride(),
					plane.offset(),
					fd,
					Some(Box::new(lease.clone())),
				)
			},
			planes => Err(err(format!(
				"cannot import a {}-plane DMA-BUF as one Vulkan plane",
				planes.len()
			))),
		};

		match direct {
			Ok(source) => Ok(Some(source)),
			#[cfg(feature = "vaapi")]
			Err(direct) => match self.retile(device, buffer, format, size, spare, lease) {
				Ok(source) => Ok(Some(source)),
				Err(retiled) => Err(err(format!("{direct}, and re-tiling it: {retiled}"))),
			},
			#[cfg(not(feature = "vaapi"))]
			Err(direct) => Err(direct),
		}
	}

	/// Blit the buffer into a driver-allocated surface and import that instead.
	///
	/// `lease` is the producer's, and is held until the blit returns: VA-API
	/// synchronizes the surface before handing it back, so once it does the
	/// pixels have been read and the producer may have its buffer again.
	#[cfg(feature = "vaapi")]
	fn retile(
		&mut self,
		device: &wgpu::Device,
		buffer: &DmaBuf,
		format: wgpu::TextureFormat,
		size: Size,
		fd: std::os::fd::OwnedFd,
		lease: std::sync::Arc<dyn crate::frame::DmaBufFrame>,
	) -> Result<Source, Error> {
		let source = moq_vaapi::vpp::DmaBuf {
			fourcc: va_fourcc(buffer.format())?,
			drm_format: buffer.format().as_raw(),
			modifier: buffer.modifier(),
			width: size.width,
			height: size.height,
			planes: buffer
				.planes()
				.iter()
				.map(|plane| moq_vaapi::vpp::Plane {
					offset: plane.offset(),
					pitch: plane.stride(),
				})
				.collect(),
			fd,
		};

		let retiled = self
			.vpp
			.get()?
			.retile(source)
			.map_err(|e| err(format!("VA-API re-tile: {e:#}")))?;
		let exported = retiled
			.export_prime()
			.map_err(|e| err(format!("export the re-tiled surface: {e}")))?;
		// The surface is only a handle on the allocation, and the exported
		// descriptor holds a reference of its own, so releasing it here leaves
		// the pixels alone. Same for the producer's buffer, now that the blit
		// has read it.
		drop(retiled);
		drop(lease);

		let moq_vaapi::DrmPrimeSurfaceDescriptor { objects, layers, .. } = exported;
		let layer = layers.first().ok_or_else(|| err("re-tiled surface has no layer"))?;
		if layer.num_planes != 1 {
			return Err(err(format!(
				"re-tile produced {} planes, which is no better than what it started from",
				layer.num_planes
			)));
		}
		let (stride, offset) = (layer.pitch[0], layer.offset[0]);
		let object = objects
			.into_iter()
			.next()
			.ok_or_else(|| err("re-tiled surface has no object"))?;

		// SAFETY: `object.fd` is the descriptor VA-API exported for a surface it
		// allocated and synchronized, and the modifier, extent, stride, and
		// offset are the ones it reported for that same surface. The allocation
		// is the post-processor's own, so no producer lease governs it.
		unsafe {
			adopt(
				device,
				format,
				size,
				object.drm_format_modifier,
				stride,
				offset,
				object.fd,
				None,
			)
		}
	}
}

/// Hand one plane's descriptor to wgpu and wrap the result as a sampled texture.
///
/// # Safety
///
/// `fd` must be a descriptor the caller owns for a DMA-BUF whose pixels are
/// written and readable, laid out exactly as `format`, `size`, `modifier`,
/// `stride`, and `offset` say. Vulkan consumes it on success and wgpu-hal closes
/// it on error, so the caller passes ownership either way.
#[expect(clippy::too_many_arguments, reason = "one DMA-BUF plane takes this many")]
unsafe fn adopt(
	device: &wgpu::Device,
	format: wgpu::TextureFormat,
	size: Size,
	modifier: u64,
	stride: u32,
	offset: u32,
	fd: std::os::fd::OwnedFd,
	keepalive: Option<Box<dyn Send + Sync>>,
) -> Result<Source, Error> {
	let extent = wgpu::Extent3d {
		width: size.width,
		height: size.height,
		depth_or_array_layers: 1,
	};
	let descriptor = wgpu::TextureDescriptor {
		label: Some("moq-video imported DMA-BUF"),
		size: extent,
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format,
		usage: wgpu::TextureUsages::TEXTURE_BINDING,
		view_formats: &[],
	};
	let hal_descriptor = wgpu::hal::TextureDescriptor {
		label: descriptor.label,
		size: extent,
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format,
		usage: wgpu::TextureUses::RESOURCE,
		memory_flags: MemoryFlags::empty(),
		view_formats: Vec::new(),
	};

	// SAFETY: the guard is only used to import a descriptor into the same
	// Vulkan device. It drops before the resulting HAL texture is wrapped.
	let hal = (unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() })
		.ok_or_else(|| err("wgpu device is not a Vulkan device"))?;
	// SAFETY: the caller's contract, forwarded.
	let texture = unsafe { hal.texture_from_dmabuf_fd(fd, &hal_descriptor, modifier, stride as u64, offset as u64) }
		.map_err(|e| err(format!("Vulkan DMA-BUF import: {e:?}")))?;
	drop(hal);

	// SAFETY: wgpu-hal created `texture` on this device from `hal_descriptor`,
	// which exactly matches the public descriptor. Imported pixels are already
	// initialized and will first be used as a sampled resource.
	let texture = unsafe {
		device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(texture, &descriptor, wgpu::TextureUses::RESOURCE)
	};
	let view = texture.create_view(&Default::default());

	Ok(Source {
		layout: Layout::Rgba,
		color: None,
		plane0: view.clone(),
		plane1: view.clone(),
		plane2: view,
		keepalive,
	})
}

/// The texture format a packed DRM format is sampled as.
fn texture_format(format: DrmFormat) -> Result<wgpu::TextureFormat, Error> {
	match format {
		DrmFormat::XRGB8888 | DrmFormat::ARGB8888 => Ok(wgpu::TextureFormat::Bgra8Unorm),
		DrmFormat::XBGR8888 | DrmFormat::ABGR8888 => Ok(wgpu::TextureFormat::Rgba8Unorm),
		format => Err(err(format!("cannot import DMA-BUF format {:#x}", format.as_raw()))),
	}
}

/// The VA-API fourcc naming the same layout as a packed DRM format.
///
/// DRM spells a pixel from its most significant byte down and VA-API from its
/// first byte in memory, so one layout has two reversed names.
#[cfg(feature = "vaapi")]
fn va_fourcc(format: DrmFormat) -> Result<u32, Error> {
	match format {
		DrmFormat::XRGB8888 => Ok(moq_vaapi::VA_FOURCC_BGRX),
		DrmFormat::ARGB8888 => Ok(moq_vaapi::VA_FOURCC_BGRA),
		DrmFormat::XBGR8888 => Ok(moq_vaapi::VA_FOURCC_RGBX),
		DrmFormat::ABGR8888 => Ok(moq_vaapi::VA_FOURCC_RGBA),
		format => Err(err(format!(
			"no VA-API format for DMA-BUF format {:#x}",
			format.as_raw()
		))),
	}
}

/// The VA-API post-processor, opened the first time a buffer needs re-tiling.
///
/// A host with no VA-API device, or one whose driver has no post-processing
/// entrypoint, is remembered as unavailable so it pays for the attempt once
/// rather than per frame.
#[cfg(feature = "vaapi")]
#[derive(Default)]
enum Vpp {
	#[default]
	Unopened,
	Ready(moq_vaapi::vpp::Processor),
	Unavailable,
}

// SAFETY: the processor is a `VADisplay` and a config id, and libva serializes
// calls on a display internally. Only the renderer owns one, only through
// `&mut self`, so it is used from one thread at a time even though which thread
// that is may change. Same argument as the Metal importer's.
#[cfg(feature = "vaapi")]
unsafe impl Send for Vpp {}

#[cfg(feature = "vaapi")]
impl Vpp {
	fn get(&mut self) -> Result<&moq_vaapi::vpp::Processor, Error> {
		if let Vpp::Unopened = self {
			*self = match moq_vaapi::vpp::Processor::open() {
				Ok(processor) => Vpp::Ready(processor),
				Err(err) => {
					tracing::debug!(%err, "no VA-API video post-processor; DMA-BUF re-tiling is unavailable");
					Vpp::Unavailable
				}
			};
		}

		match self {
			Vpp::Ready(processor) => Ok(processor),
			_ => Err(err("no VA-API video post-processor on this host")),
		}
	}
}
