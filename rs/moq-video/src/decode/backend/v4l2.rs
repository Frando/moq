//! Hardware H.264 decode via the V4L2 stateful M2M decoder (Linux).
//!
//! The mirror of the encode backend, on the same device abstraction and usually
//! on a sibling node of the same driver: a Raspberry Pi decodes on
//! `/dev/video10` and encodes on `/dev/video11`, both `bcm2835-codec`. Behind
//! the same non-default `v4l2` feature.
//!
//! Access units go in on the OUTPUT queue in decode order and pictures come back
//! on CAPTURE as CPU [`Surface::I420`](crate::Surface). The driver copies each
//! input buffer's timestamp onto the picture it produced, so presentation times
//! survive decoder delay without any bookkeeping here.
//!
//! Stateful is the operative word. The driver parses the stream itself and
//! announces the picture size with a `V4L2_EVENT_SOURCE_CHANGE`, which is why
//! the CAPTURE queue does not exist until the first parameter sets have been
//! fed: [`Backend::decode`] returns no frames until it arrives, which is the
//! buffering the trait already allows for. The same event later means the stream
//! changed size, and the CAPTURE queue is torn down and renegotiated.
//!
//! Only H.264, and only stateful. A Raspberry Pi 4's separate HEVC block and
//! Rockchip's `rkvdec` are *stateless* V4L2 decoders: they take per-slice
//! parameters through the media request API and expect userspace to have parsed
//! the bitstream, which is a different interface rather than another format
//! here. [`Config::resize`] is ignored, since these drivers scale on a separate
//! ISP node rather than on the decoder.
//!
//! NOT YET VALIDATED ON HARDWARE: compile-verified only, on a machine with no
//! M2M codec device. The ioctl sequence follows the kernel's stateful decoder
//! documentation and an implementation that ran on Pi Zero 2 W / Pi 3 / Pi 4,
//! but this port needs a Pi to confirm.

use std::time::{Duration, Instant};

use bytes::Bytes;
use moq_net::Timestamp;
use v4l::v4l_sys::V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

use super::{Backend, Codec, Config};
use crate::v4l2::{self, Dequeue, Device, Dir, Format, Planes, Queue, Request, Role};
use crate::{Error, Frame, Size, Surface};

pub(crate) const NAME: &str = "v4l2";

/// Which node decodes: one that takes H.264 in and gives raw 4:2:0 back.
const ROLE: Role = Role {
	env: "MOQ_V4L2_DECODER",
	input: &[v4l2::H264],
	output: v4l2::RAW,
};

/// Access units the driver may hold at once.
const CODED_BUFFERS: u32 = 4;

/// Bytes to reserve per access unit. The catalog does not reach a backend, so
/// there is no resolution to size this from, and one megabyte clears a 1080p
/// keyframe at any bitrate worth sending over a network.
const CODED_SIZE: u32 = 1024 * 1024;

/// Pictures to allocate beyond the driver's stated minimum, so a caller holding
/// a frame or two does not stall decoding.
const SPARE_PICTURES: u32 = 3;

/// Pictures to allocate when the driver will not say what it needs.
const DEFAULT_PICTURES: u32 = 8;

/// How long [`Backend::decode`] waits for the driver to take an access unit
/// before giving up on it.
const BUFFER_TIMEOUT: Duration = Duration::from_millis(500);

/// How long a call waits for the source change that follows the first parameter
/// sets. Missing it is not an error on its own: a subscriber can join anywhere
/// in a group, so the first access units of a stream may well be undecodable
/// and the next one gets another look.
const SOURCE_CHANGE_TIMEOUT: Duration = Duration::from_millis(250);

/// How long the decoder spends waiting for that source change in total before
/// giving up on the stream.
///
/// Without a ceiling, a stream the driver can never size costs every call the
/// whole of [`SOURCE_CHANGE_TIMEOUT`] and returns nothing, which is a permanent
/// four frames a second with no error and no log. Generous enough to cover a
/// join a long way from the next keyframe.
const SOURCE_CHANGE_BUDGET: Duration = Duration::from_secs(5);

/// How long a resolution change waits for the driver to hand back the pictures
/// it decoded before it. Missing the tail costs those pictures but not the
/// stream, so it is bounded rather than waited out.
const DRAIN_TIMEOUT: Duration = Duration::from_millis(500);

/// How long each wait inside those loops parks for.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

pub(crate) struct V4l2 {
	device: Device,
	/// The OUTPUT queue: access units going in.
	coded: Queue,
	/// The CAPTURE queue, which exists only once the driver has reported the
	/// stream's size.
	pictures: Option<Pictures>,
	/// When the first access unit went in, which is what
	/// [`SOURCE_CHANGE_BUDGET`] is measured from.
	since: Option<Instant>,
}

/// The CAPTURE queue plus where a picture sits in one of its buffers.
struct Pictures {
	queue: Queue,
	planes: Planes,
}

impl V4l2 {
	pub(crate) fn open(codec: Codec, _config: &Config) -> Result<Box<dyn Backend>, Error> {
		if codec != Codec::H264 {
			return Err(Error::UnsupportedCodec(format!("{NAME} decodes H.264 only")));
		}

		let device = v4l2::open(&ROLE)?;
		// Before the first access unit, so the announcement of the stream's size
		// cannot be missed.
		device.subscribe_source_change()?;

		let coded = device.set_format(
			Dir::Output,
			&Request {
				pixelformat: v4l2::H264,
				// The driver reads the real dimensions out of the bitstream. This is
				// only a hint for how large a coded buffer to allocate, which the
				// explicit `sizeimage` overrides anyway.
				size: Size::new(1920, 1088),
				sizeimage: Some(CODED_SIZE),
				color: None,
			},
		)?;
		if coded.pixelformat != v4l2::H264 {
			return Err(Error::Codec(anyhow::anyhow!(
				"V4L2 decoder answered an H264 request with {}",
				v4l2::name(coded.pixelformat)
			)));
		}

		let mut coded = Queue::alloc(&device, Dir::Output, coded, CODED_BUFFERS)?;
		// Started immediately, unlike the encoder's: a stateful decoder has to be
		// consuming before it can parse a stream and report its size.
		coded.stream_on(&device)?;

		tracing::info!(
			decoder = NAME,
			device = %device.path().display(),
			"opened H.264 decoder"
		);

		Ok(Box::new(Self {
			device,
			coded,
			pictures: None,
			since: None,
		}))
	}

	/// Hand one access unit to the driver, waiting for a free buffer.
	fn submit(&mut self, access_unit: &Bytes, timestamp: Timestamp) -> Result<(), Error> {
		let deadline = Instant::now() + BUFFER_TIMEOUT;
		let index = loop {
			while let Some(buffer) = self.coded.dequeue(&self.device)?.buffer() {
				if buffer.failed() {
					tracing::warn!(
						decoder = NAME,
						buffer = buffer.index,
						"V4L2 decoder could not decode an access unit"
					);
				}
				self.coded.reclaim(buffer.index);
			}
			if let Some(index) = self.coded.take_free() {
				break index;
			}
			if Instant::now() >= deadline {
				return Err(Error::Codec(anyhow::anyhow!(
					"V4L2 decoder held every input buffer for {BUFFER_TIMEOUT:?}"
				)));
			}
			self.device.wait(POLL_INTERVAL);
		};

		let capacity = self.coded.plane(index, 0).len();
		if access_unit.len() > capacity {
			// The buffer goes back on the free list: losing one to a single oversized
			// access unit would shrink the pool for the rest of the stream.
			self.coded.reclaim(index);
			return Err(Error::Codec(anyhow::anyhow!(
				"access unit of {} bytes exceeds the V4L2 decoder's {capacity} byte buffer",
				access_unit.len()
			)));
		}
		self.coded.plane_mut(index, 0)[..access_unit.len()].copy_from_slice(access_unit);

		let bytesused = [access_unit.len() as u32];
		let timestamp = Duration::from_micros(timestamp.as_micros() as u64);
		self.coded.queue(&self.device, index, &bytesused, timestamp)
	}

	/// Negotiate the CAPTURE queue against the size the driver has just reported.
	fn negotiate(&mut self) -> Result<(), Error> {
		if let Some(pictures) = self.pictures.take() {
			pictures.queue.release(&self.device)?;
		}

		let format = self.capture_format()?;
		// Read after the format is settled, since `VIDIOC_S_FMT` on CAPTURE resets
		// the compose rectangle to the default for the format it just took.
		//
		// The coded size is rounded up to whole macroblocks, so the picture is the
		// compose rectangle inside it. A driver that reports none codes exactly the
		// picture.
		let size = self.device.visible_size(Dir::Capture).unwrap_or(format.size);
		let planes = Planes::new(&format, size)?;

		// The driver needs a minimum of its own to hold reference frames; anything
		// below it decodes wrong or not at all.
		let minimum = self
			.device
			.control(V4L2_CID_MIN_BUFFERS_FOR_CAPTURE)
			.map_or(DEFAULT_PICTURES, |minimum| minimum.max(1) as u32 + SPARE_PICTURES);

		tracing::info!(
			decoder = NAME,
			format = v4l2::name(format.pixelformat),
			coded = %format.size,
			visible = %size,
			buffers = minimum,
			"V4L2 decoder negotiated its output"
		);

		let mut queue = Queue::alloc(&self.device, Dir::Capture, format, minimum)?;
		while let Some(index) = queue.take_free() {
			queue.queue(&self.device, index, &[0; 3], Duration::ZERO)?;
		}
		queue.stream_on(&self.device)?;

		self.pictures = Some(Pictures { queue, planes });
		Ok(())
	}

	/// Choose the raw format the CAPTURE queue produces.
	///
	/// `VIDIOC_G_FMT` reports the driver's own default, which need not be one this
	/// code can lay out: amphion defaults to the tiled `NV12_8L128` and mtk-vcodec
	/// to `MM21`. That the node offered NV12 at open does not settle it either,
	/// since the set narrows to what the driver supports for the stream it has now
	/// parsed. `VIDIOC_ENUM_FMT` re-reads that set and `VIDIOC_S_FMT` is the step
	/// where userspace picks from it. See
	/// `Documentation/userspace-api/media/v4l/dev-decoder.rst`, "Capture Setup",
	/// steps 3 and 4.
	fn capture_format(&self) -> Result<Format, Error> {
		let format = self.device.format(Dir::Capture)?;
		if v4l2::RAW.contains(&format.pixelformat) {
			return Ok(format);
		}

		let offered = self.device.formats(Dir::Capture)?;
		let Some(&pixelformat) = v4l2::RAW.iter().find(|code| offered.contains(code)) else {
			return Err(Error::Codec(anyhow::anyhow!(
				"V4L2 decoder defaults to {} and offers no 8-bit 4:2:0 format for this stream",
				v4l2::name(format.pixelformat)
			)));
		};

		tracing::debug!(
			decoder = NAME,
			default = v4l2::name(format.pixelformat),
			selected = v4l2::name(pixelformat),
			"V4L2 decoder defaulted to a format this cannot read"
		);
		self.device.set_format(
			Dir::Capture,
			&Request {
				pixelformat,
				// Unchanged from what the driver parsed out of the stream: this has no
				// compose or scaling to ask for.
				size: format.size,
				sizeimage: None,
				color: None,
			},
		)
	}

	/// Collect every picture the driver has finished, re-queueing each buffer as
	/// it is copied out.
	///
	/// Returns whether the sequence ended, which the driver says with
	/// `V4L2_BUF_FLAG_LAST` on its last picture and then with `EPIPE` on any
	/// dequeue past it.
	fn drain(&mut self, frames: &mut Vec<Frame>) -> Result<bool, Error> {
		let Some(pictures) = &self.pictures else {
			return Ok(false);
		};

		loop {
			let buffer = match pictures.queue.dequeue(&self.device)? {
				Dequeue::Buffer(buffer) => buffer,
				Dequeue::Empty => return Ok(false),
				Dequeue::Ended => return Ok(true),
			};

			// A zero-length picture is the driver marking the end of a sequence
			// rather than a frame, which is what arrives just before a source change.
			let decoded = match buffer.bytesused[0] {
				0 => None,
				// A buffer flagged `V4L2_BUF_FLAG_ERROR` dequeues successfully and holds
				// a picture the driver could not decode, so reading it out would publish
				// garbage under a valid timestamp.
				_ if buffer.failed() => {
					tracing::warn!(
						decoder = NAME,
						buffer = buffer.index,
						"V4L2 decoder flagged a picture bad"
					);
					None
				}
				_ => Some(pictures.planes.read(&pictures.queue, buffer.index)?),
			};
			// Back to the driver before anything can fail: a picture buffer left out
			// of the pool is one the decoder never gets to write again.
			pictures
				.queue
				.queue(&self.device, buffer.index, &[0; 3], Duration::ZERO)?;

			if let Some(decoded) = decoded {
				let timestamp = Timestamp::from_micros(buffer.timestamp.as_micros() as u64)?;
				frames.push(Frame::new(Surface::I420(decoded), timestamp));
			}
			if buffer.last() {
				return Ok(true);
			}
		}
	}

	/// Take the pictures still in the CAPTURE queue when a sequence ends.
	///
	/// A source change is an implicit drain: the driver decodes everything from
	/// before the change, marks the last of it with `V4L2_BUF_FLAG_LAST`, and only
	/// then is the CAPTURE queue free to be torn down. Releasing it as soon as the
	/// event arrives discards every picture the driver had already decoded, which
	/// is a visible gap at each mid-stream resolution change. See
	/// `Documentation/userspace-api/media/v4l/dev-decoder.rst`, "Dynamic Resolution
	/// Change".
	fn drain_tail(&mut self) -> Result<Vec<Frame>, Error> {
		let mut frames = Vec::new();
		let deadline = Instant::now() + DRAIN_TIMEOUT;
		loop {
			if self.drain(&mut frames)? {
				return Ok(frames);
			}
			if Instant::now() >= deadline {
				// Renegotiated anyway: a tail the driver never finished is worth less
				// than the rest of the stream, and holding the old queue open does not
				// make it arrive.
				tracing::warn!(
					decoder = NAME,
					frames = frames.len(),
					"V4L2 decoder did not end the sequence within {DRAIN_TIMEOUT:?}"
				);
				return Ok(frames);
			}
			self.device.wait(POLL_INTERVAL);
		}
	}
}

impl Backend for V4l2 {
	fn decode(&mut self, access_unit: Bytes, timestamp: Timestamp, _keyframe: bool) -> Result<Vec<Frame>, Error> {
		self.submit(&access_unit, timestamp)?;

		// Before the first negotiation there is nowhere for a picture to go, so the
		// event is worth waiting for. After it, checking costs one ioctl and a
		// missed resolution change would decode the rest of the stream at the old
		// geometry.
		let mut frames = Vec::new();
		if self.pictures.is_none() {
			let since = *self.since.get_or_insert_with(Instant::now);
			let deadline = Instant::now() + SOURCE_CHANGE_TIMEOUT;
			while !self.device.take_source_change() {
				if Instant::now() >= deadline {
					let waited = since.elapsed();
					if waited >= SOURCE_CHANGE_BUDGET {
						return Err(Error::Codec(anyhow::anyhow!(
							"V4L2 decoder did not report the stream's size within {waited:?}"
						)));
					}
					tracing::debug!(
						decoder = NAME,
						?waited,
						"V4L2 decoder has not reported the stream's size"
					);
					return Ok(Vec::new());
				}
				self.device.wait(POLL_INTERVAL);
			}
			self.negotiate()?;
		} else if self.device.take_source_change() {
			// The pictures decoded before the change are still in the CAPTURE queue,
			// and the kernel wants them taken before the queue is released.
			frames = self.drain_tail()?;
			self.negotiate()?;
		}

		self.drain(&mut frames)?;
		Ok(frames)
	}

	fn name(&self) -> &str {
		NAME
	}
}
