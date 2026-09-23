//! VA-API's side of [`DmaBuf`](super::DmaBuf): handing one to `moq-vaapi`,
//! wrapping what `moq-vaapi` exports, and scaling one on the GPU.
//!
//! The decoder and the encoder backends and
//! [`Surface::resize`](super::Surface::resize) all meet
//! `moq-vaapi` here, so there is one place that knows how a crate `DmaBuf` maps
//! to a VA-API import descriptor and back.
//!
//! Scaling goes through the VA-API video processor on whichever device answers
//! first. It stays on the GPU: the input is imported, blitted into an NV12
//! surface of the target size (converting packed RGB on the way), and that
//! surface is exported as a new DMA-BUF, which the VAAPI encoder then imports
//! in turn. A simulcast publisher resizing one captured frame into several
//! renditions pays one blit per rendition instead of a download and a CPU
//! scale each.

use std::cell::RefCell;
use std::collections::HashSet;
use std::os::fd::{AsFd, OwnedFd};
use std::sync::{Arc, Mutex};

use moq_vaapi::decode::ExportedFrame;
use moq_vaapi::dmabuf::{DmaBuf as VaapiDmaBuf, Plane};
use moq_vaapi::vpp::Processor;
use moq_vaapi::{Matrix, VA_FOURCC_NV12};

use super::{DmaBuf, DmaBufExport, DmaBufFrame, DmaBufPlane, DrmFormat, I420};
use crate::{Color, Error, Size};

/// Describes `buffer` for a VA-API import, returning the producer lease to hold until the import is done with.
///
/// Waits on the producer's write fence first, since a VA-API import does not.
/// The descriptor handed to `moq-vaapi` is a duplicate, owned by the imported
/// surface; the returned lease keeps the producer from recycling the buffer
/// underneath it.
pub(crate) fn import(buffer: &DmaBuf) -> Result<(VaapiDmaBuf, DmaBufExport), Error> {
	let export = buffer
		.export()
		.map_err(|e| Error::Codec(anyhow::anyhow!("export a DMA-BUF for VA-API: {e}")))?;
	let fd: OwnedFd = export
		.as_fd()
		.try_clone_to_owned()
		.map_err(|e| Error::Codec(anyhow::anyhow!("duplicate a DMA-BUF descriptor: {e}")))?;
	let descriptor = VaapiDmaBuf {
		drm_format: buffer.format().as_raw(),
		modifier: buffer.modifier(),
		width: buffer.width(),
		height: buffer.height(),
		planes: buffer
			.planes()
			.iter()
			.map(|plane| Plane {
				offset: plane.offset(),
				pitch: plane.stride(),
			})
			.collect(),
		fd,
		color: buffer.color().map(color),
	};
	Ok((descriptor, export))
}

/// Returns the `moq-vaapi` color space matching `color`.
pub(crate) fn color(color: Color) -> moq_vaapi::Color {
	match color {
		Color::Bt601Limited => moq_vaapi::Color {
			matrix: Matrix::Bt601,
			full_range: false,
		},
		Color::Bt601Full => moq_vaapi::Color {
			matrix: Matrix::Bt601,
			full_range: true,
		},
		Color::Bt709Limited => moq_vaapi::Color {
			matrix: Matrix::Bt709,
			full_range: false,
		},
		Color::Bt709Full => moq_vaapi::Color {
			matrix: Matrix::Bt709,
			full_range: true,
		},
	}
}

/// Returns whether `format` is packed RGB, which carries no YUV matrix of its own.
pub(crate) fn is_rgb(format: DrmFormat) -> bool {
	matches!(
		format,
		DrmFormat::XRGB8888 | DrmFormat::ARGB8888 | DrmFormat::XBGR8888 | DrmFormat::ABGR8888
	)
}

thread_local! {
	/// The video processor this thread scales with, opened on the first resize.
	///
	/// Per thread because a processor keeps its contexts behind an `Rc`. Each
	/// thread that resizes pays for one display and one context per size. A
	/// failed open is kept too, so a machine without VA-API finds that out once
	/// per thread rather than once per frame.
	static PROCESSOR: RefCell<Option<Result<Processor, String>>> = const { RefCell::new(None) };

	/// Buffer layouts the video processor refused on this thread, by format and
	/// modifier, so a buffer it cannot read goes straight to the CPU path.
	static REFUSED: RefCell<HashSet<(DrmFormat, u64)>> = RefCell::new(HashSet::new());
}

/// Scales `buffer` to `size` on the GPU, returning an NV12 DMA-BUF.
///
/// The color handling matches the CPU resize. A YUV input keeps whatever space
/// it was labelled with, including none. A packed RGB input is converted into
/// [`Color::infer`] for its own size, the space the CPU download would have
/// picked, and the result is labelled with it.
///
/// # Errors
///
/// Fails when this thread has no video processor, when the processor refused
/// this buffer's format and modifier before, and when the export, the import,
/// or the blit fails. The caller falls back to the CPU.
pub(crate) fn resize(buffer: &DmaBuf, size: Size) -> Result<DmaBuf, Error> {
	let key = (buffer.format(), buffer.modifier());
	if REFUSED.with(|refused| refused.borrow().contains(&key)) {
		return Err(Error::Unsupported(format!(
			"the VA-API video processor refused {:?} with modifier {:#x}",
			key.0, key.1
		)));
	}
	let (input_space, output_space, label) = match is_rgb(buffer.format()) {
		true => {
			let inferred = Color::infer(Size::new(buffer.width(), buffer.height()));
			(None, Some(color(inferred)), Some(inferred))
		}
		false => (buffer.color().map(color), buffer.color().map(color), buffer.color()),
	};

	let exported = PROCESSOR.with(|slot| -> Result<ExportedFrame, Error> {
		let mut slot = slot.borrow_mut();
		// Checked before exporting, so a machine without VA-API does not pay a
		// fence wait and a descriptor duplicate on every frame to learn it again.
		let processor = slot
			.get_or_insert_with(|| Processor::open().map_err(|e| format!("{e:#}")))
			.as_ref()
			.map_err(|e| Error::Unsupported(format!("no VA-API video processor: {e}")))?;
		let (descriptor, lease) = import(buffer)?;
		let processed = processor.import(descriptor).and_then(|input| {
			processor.process(
				&input,
				VA_FOURCC_NV12,
				(size.width, size.height),
				input_space,
				output_space,
			)
		});
		// The blit has synced, so the producer's buffer is free to go back.
		drop(lease);
		let output = processed.map_err(|e| {
			REFUSED.with(|refused| refused.borrow_mut().insert(key));
			Error::Codec(e.context("scale a DMA-BUF with the VA-API video processor"))
		})?;
		ExportedFrame::from_surface(output, 0).map_err(Error::Codec)
	})?;
	adopt(exported, label).map_err(Error::Codec)
}

/// Describes an exported picture as a [`DmaBuf`]: the driver's format modifier,
/// and the offset and pitch of each of its memory planes.
///
/// The width and height are the visible frame rather than the exported extent,
/// which is the driver's padded allocation. Neither the pitches nor the offsets
/// follow from the visible size, which is exactly why they are read off the
/// export rather than computed from it. `color` is the space the pixels are in,
/// where the producer knows it.
///
/// # Errors
///
/// When the export is not the one shape a [`DmaBuf`] can describe: a single NV12
/// layer whose planes all live in a single object. The Intel and AMD drivers
/// export exactly that, and the alternatives are refused rather than guessed at,
/// because every one of them draws as a plausible-looking picture made of the
/// wrong bytes.
pub(crate) fn adopt(frame: ExportedFrame, color: Option<Color>) -> anyhow::Result<DmaBuf> {
	let (width, height) = (frame.width, frame.height);
	// One object, because a consumer imports every plane from the one descriptor
	// `Exported::export` hands out, and one layer, because the planes are read
	// off it as a group. Both are what `VA_EXPORT_SURFACE_COMPOSED_LAYERS` asks
	// for; neither is what it guarantees.
	let [object] = frame.descriptor.objects.as_slice() else {
		anyhow::bail!(
			"VA-API exported {} objects, expected one holding every plane",
			frame.descriptor.objects.len()
		);
	};
	let [layer] = frame.descriptor.layers.as_slice() else {
		anyhow::bail!(
			"VA-API exported {} layers, expected one composed layer",
			frame.descriptor.layers.len()
		);
	};
	if layer.drm_format != DrmFormat::NV12.as_raw() {
		anyhow::bail!("VA-API exported DRM format {:#x}, expected NV12", layer.drm_format);
	}

	// `num_planes` and the arrays it indexes both come from the driver, and only
	// the arrays are bounded, so indexing on the count would panic rather than
	// fail.
	let count = layer.num_planes as usize;
	anyhow::ensure!(
		count <= layer.offset.len(),
		"VA-API exported {count} planes, more than a PRIME descriptor holds"
	);
	let planes = (0..count)
		.map(|plane| DmaBufPlane::new(layer.offset[plane], layer.pitch[plane]))
		.collect();
	let modifier = object.drm_format_modifier;

	DmaBuf::new(
		DrmFormat::NV12,
		modifier,
		width,
		height,
		planes,
		color,
		Arc::new(Exported {
			frame: Mutex::new(frame),
			color,
		}),
	)
	.map_err(|e| anyhow::anyhow!("{e}"))
}

/// A picture on a VA-API surface the consumer holds as a DMA-BUF.
///
/// Both of the things a consumer can do with one: hand a descriptor to a
/// graphics API, or give up on drawing it and read the pixels back. Dropping the
/// last clone destroys the surface, which is what returns its allocation to the
/// driver.
struct Exported {
	/// Locked because [`DmaBufFrame`] hands out `&self` while what is behind it
	/// is a single libva surface: `download_i420` maps that surface, and two
	/// threads doing so at once is more than libva promises to serialize. The
	/// frame is [`Send`] on its own, so a lock is enough and no `unsafe impl` is
	/// involved.
	frame: Mutex<ExportedFrame>,
	/// The space the pixels are in, for the read-back to carry.
	color: Option<Color>,
}

impl DmaBufFrame for Exported {
	/// Vulkan takes ownership of an imported descriptor on success and closes it
	/// on failure, so every import needs one of its own and the original stays
	/// with the picture.
	fn export(&self) -> std::io::Result<OwnedFd> {
		let frame = self.frame.lock().expect("poisoned");
		let object = frame.descriptor.objects.first().ok_or_else(|| {
			std::io::Error::new(std::io::ErrorKind::InvalidData, "the VA-API export carries no object")
		})?;
		object.fd.as_fd().try_clone_to_owned()
	}

	/// Read the picture back through the retained surface rather than the
	/// descriptor: a VA-API surface is tiled, so mapping the file descriptor as
	/// rows would be wrong.
	fn download_i420(&self) -> Result<I420, Error> {
		let frame = self.frame.lock().expect("poisoned");
		let nv12 = frame
			.download()
			.map_err(|e| Error::Codec(anyhow::anyhow!("read a VA-API surface back: {e:?}")))?;
		let i420 = I420::from_nv12(&nv12.data, crate::Size::new(nv12.width, nv12.height))?;
		Ok(match self.color {
			Some(color) => i420.with_color(color),
			None => i420,
		})
	}
}

// For the encoder backend's tests, which need openh264 to decode.
#[cfg(all(test, feature = "openh264"))]
pub(crate) mod testing {
	use std::os::fd::OwnedFd;
	use std::sync::Arc;

	use moq_vaapi::{Display, Image, Surface as VaSurface, UsageHint, VA_FOURCC_BGRX, VA_RT_FORMAT_RGB32};

	use super::super::{DmaBuf, DmaBufFrame, DmaBufPlane, DrmFormat, I420};
	use crate::{Error, Size};

	/// A driver surface exported as a DMA-BUF, standing in for a PipeWire buffer.
	struct Allocated {
		_surface: VaSurface<()>,
		fd: OwnedFd,
	}

	impl DmaBufFrame for Allocated {
		fn export(&self) -> std::io::Result<OwnedFd> {
			self.fd.try_clone()
		}

		fn download_i420(&self) -> Result<I420, Error> {
			Err(Error::Unsupported("a test buffer has no CPU path".into()))
		}
	}

	/// A buffer no driver imports, whose CPU read-back still works.
	struct Unimportable {
		fd: OwnedFd,
		pixels: I420,
	}

	impl DmaBufFrame for Unimportable {
		fn export(&self) -> std::io::Result<OwnedFd> {
			self.fd.try_clone()
		}

		fn download_i420(&self) -> Result<I420, Error> {
			Ok(self.pixels.clone())
		}
	}

	/// Returns an NV12 DMA-BUF the driver refuses to import (its descriptor is
	/// `/dev/null` and its modifier made up) whose CPU read-back yields `pixels`.
	pub(crate) fn unimportable_dmabuf(pixels: I420) -> DmaBuf {
		let fd = OwnedFd::from(std::fs::File::open("/dev/null").expect("open /dev/null"));
		let (width, height) = (pixels.width(), pixels.height());
		DmaBuf::new(
			DrmFormat::NV12,
			0x00ff_ffff_ffff_fffe,
			width,
			height,
			vec![DmaBufPlane::new(0, width), DmaBufPlane::new(width * height, width)],
			None,
			Arc::new(Unimportable { fd, pixels }),
		)
		.expect("a valid description")
	}

	/// Returns a BGRX DMA-BUF holding `rgba`, allocated by VA-API, or `None` without a device.
	pub(crate) fn bgrx_dmabuf(rgba: &[u8], size: Size) -> Option<DmaBuf> {
		let display = Display::open()?;
		let (width, height) = (size.width, size.height);
		let surface = display
			.create_surfaces(
				VA_RT_FORMAT_RGB32,
				Some(VA_FOURCC_BGRX),
				width,
				height,
				Some(UsageHint::USAGE_HINT_EXPORT),
				vec![()],
			)
			.ok()?
			.pop()?;
		let format = display
			.query_image_formats()
			.ok()?
			.into_iter()
			.find(|format| format.fourcc == VA_FOURCC_BGRX)?;
		{
			let mut image = Image::create_from(&surface, format, (width, height), (width, height)).ok()?;
			let va_image = *image.image();
			let data = image.as_mut();
			for y in 0..height as usize {
				for x in 0..width as usize {
					let from = (y * width as usize + x) * 4;
					let to = va_image.offsets[0] as usize + y * va_image.pitches[0] as usize + x * 4;
					let [r, g, b, _] = [rgba[from], rgba[from + 1], rgba[from + 2], rgba[from + 3]];
					data[to..to + 4].copy_from_slice(&[b, g, r, 255]);
				}
			}
		}
		surface.sync().ok()?;
		let mut exported = surface.export_prime().ok()?;
		let layer = &exported.layers[0];
		assert_eq!(layer.drm_format, DrmFormat::XRGB8888.as_raw());
		let plane = DmaBufPlane::new(layer.offset[0], layer.pitch[0]);
		let object = exported.objects.remove(0);
		DmaBuf::new(
			DrmFormat::XRGB8888,
			object.drm_format_modifier,
			width,
			height,
			vec![plane],
			None,
			Arc::new(Allocated {
				_surface: surface,
				fd: object.fd,
			}),
		)
		.ok()
	}
}
