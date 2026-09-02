//! Hardware H.264 backend via the V4L2 stateful M2M encoder (Linux).
//!
//! The encoder most ARM SoCs ship: a Raspberry Pi's VideoCore (`bcm2835-codec`),
//! and the equivalent block on Rockchip, Amlogic, Allwinner, and Samsung parts.
//! Without it a Pi either republishes what `rpicam-vid` already encoded or
//! spends its CPU on openh264, so this is the difference between a Pi Zero
//! publishing 1080p and not publishing at all.
//!
//! Behind the non-default `v4l2` feature. It costs no runtime dependency (the
//! interface is ioctls on a device node), only the `v4l` crate's build-time
//! bindgen, and a host with no M2M node fails at open so automatic selection
//! falls through to the next encoder.
//!
//! Feeds the driver 8-bit 4:2:0 and takes back Annex-B with in-band SPS/PPS
//! ahead of every IDR, which is the avc3 shape the H.264 importer expects.
//! [`Planes`](crate::v4l2::Planes) owns the layout: the driver picks the raw
//! fourcc, the stride, and the padded row count, and none of the three has to be
//! what was asked for.
//!
//! Two things the driver will not do by itself, both learned on a Pi:
//!   1. `bcm2835-codec` defaults to H.264 level 1.0, which is 128x96. The level
//!      has to be set from the resolution before `VIDIOC_S_FMT` or the encoder
//!      refuses anything larger, and no ioctl reports the default, so there is
//!      nothing to detect and the level is always set.
//!   2. Repeated parameter sets have two controls and drivers implement one
//!      each: `bcm2835-codec` has `REPEAT_SEQ_HEADER` and not
//!      `PREPEND_SPSPPS_TO_IDR`. Both are asked for and whichever lands wins,
//!      because a subscriber joining at a later keyframe can only start if that
//!      keyframe carries SPS/PPS.
//!
//! NOT YET VALIDATED ON HARDWARE: compile-verified only. The ioctl sequence is
//! carried over from an implementation that ran on Pi Zero 2 W / Pi 3 / Pi 4,
//! but this port has not been run on a Pi, so the emitted bitstream needs one to
//! confirm at playback.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use moq_net::Timestamp;
use v4l::v4l_sys::{
	V4L2_CID_MPEG_VIDEO_BITRATE, V4L2_CID_MPEG_VIDEO_BITRATE_MODE, V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME,
	V4L2_CID_MPEG_VIDEO_GOP_SIZE, V4L2_CID_MPEG_VIDEO_H264_LEVEL, V4L2_CID_MPEG_VIDEO_H264_PROFILE,
	V4L2_CID_MPEG_VIDEO_HEADER_MODE, V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR, V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER,
	V4L2_ENC_CMD_START, V4L2_ENC_CMD_STOP, v4l2_mpeg_video_bitrate_mode_V4L2_MPEG_VIDEO_BITRATE_MODE_CBR,
	v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_1_3,
	v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_3_0,
	v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_3_1,
	v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_4_0,
	v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_5_1,
	v4l2_mpeg_video_h264_profile_V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE,
	v4l2_mpeg_video_header_mode_V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME,
};

use super::super::encoder::Config;
use super::{Backend, Encoded};
use crate::v4l2::{self, Dequeue, Device, Dir, Planes, Queue, Request, Role};
use crate::{Error, Frame, Size};

pub(crate) const NAME: &str = "v4l2";

/// Which node encodes: one that takes raw 4:2:0 in and gives H.264 back.
const ROLE: Role = Role {
	env: "MOQ_V4L2_ENCODER",
	input: v4l2::RAW,
	output: &[v4l2::H264],
};

/// Raw frames the driver may hold at once. Enough to keep the hardware fed
/// while the caller converts the next frame, and no more: every buffer is a
/// full picture of memory on a board that has little.
const RAW_BUFFERS: u32 = 4;

/// Coded buffers. More than [`RAW_BUFFERS`] so a run of large access units
/// never leaves the encoder with nowhere to write.
const CODED_BUFFERS: u32 = 8;

/// How long [`Backend::encode`] waits for the driver to hand a raw buffer back
/// before giving up on the frame. Reached only if the hardware has stalled:
/// with [`RAW_BUFFERS`] outstanding, an encoder keeping up returns one within a
/// frame interval.
const BUFFER_TIMEOUT: Duration = Duration::from_millis(500);

/// How long [`Backend::flush`] waits for the codec to hand back an access unit
/// for a frame it has already taken.
const FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

/// How long each wait inside those loops parks for. Short enough that the
/// timeouts above are honored closely, long enough not to spin.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

pub(crate) struct V4l2 {
	device: Device,
	/// The OUTPUT queue: raw frames going in.
	raw: Queue,
	/// The CAPTURE queue: coded access units coming back.
	coded: Queue,
	/// Where a raw frame's samples go in a `raw` buffer.
	planes: Planes,
	size: Size,
	/// Frames the driver has taken and not yet answered with an access unit.
	pending: Pending,
	/// Whether the driver takes `VIDIOC_ENCODER_CMD`. Cleared the first time it
	/// refuses one, after which a flush can only take what the codec has already
	/// finished.
	drainable: bool,
	/// Whether the driver takes `V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME`. Cleared
	/// the first time it refuses one, which is worth saying once and not per
	/// keyframe.
	keyframes: bool,
}

impl V4l2 {
	pub(crate) fn open(config: &Config) -> Result<Box<dyn Backend>, Error> {
		let size = config.size();
		size.validate("V4L2 encode of")?;

		let device = v4l2::open(&ROLE)?;

		// Before `VIDIOC_S_FMT`: bcm2835-codec picks its encoder configuration
		// from the profile and level at format time, and a level set afterwards
		// does not take.
		device.set_control(
			V4L2_CID_MPEG_VIDEO_H264_PROFILE,
			v4l2_mpeg_video_h264_profile_V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE as i32,
		)?;
		device.set_control(V4L2_CID_MPEG_VIDEO_H264_LEVEL, h264_level(size))?;
		device.set_control(V4L2_CID_MPEG_VIDEO_GOP_SIZE, config.gop as i32)?;
		set_bitrate(&device, config.resolved_bitrate())?;
		// Constant rate is what a live uplink wants: the congestion controller
		// already owns the rate, and a variable-rate encoder would spend its
		// budget on the wrong frames.
		device.try_control(
			V4L2_CID_MPEG_VIDEO_BITRATE_MODE,
			v4l2_mpeg_video_bitrate_mode_V4L2_MPEG_VIDEO_BITRATE_MODE_CBR as i32,
		);

		// The parameter sets belong in the first access unit rather than on a coded
		// buffer of their own. A driver left in `HEADER_MODE_SEPARATE`, which is
		// s5p-mfc's default, emits them matched to no source frame and with no
		// timestamp worth publishing them under. Optional because not every driver
		// has the control, and [`Pending`] carries them into the next access unit
		// where it does not land.
		device.try_control(
			V4L2_CID_MPEG_VIDEO_HEADER_MODE,
			v4l2_mpeg_video_header_mode_V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME as i32,
		);

		// The two spellings of "repeat the parameter sets", neither of which every
		// driver has. A subscriber joining at any keyframe needs SPS/PPS in band
		// ahead of it, so if neither lands the track is only joinable at its first
		// keyframe and that is worth a warning.
		let repeat = device.try_control(V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER, 1)
			| device.try_control(V4L2_CID_MPEG_VIDEO_PREPEND_SPSPPS_TO_IDR, 1);
		if !repeat {
			tracing::warn!(
				encoder = NAME,
				device = %device.path().display(),
				"driver repeats no parameter sets; subscribers can only join at the first keyframe"
			);
		}

		// Coded first, raw second, which is the order
		// `Documentation/userspace-api/media/v4l/dev-encoder.rst` gives under
		// "Initialization" and not just a preference: an encoder derives a new
		// OUTPUT format from the CAPTURE format it is given, so a raw format
		// negotiated first is one the driver is free to have replaced by the time
		// its stride is read.
		let coded = device.set_format(
			Dir::Capture,
			&Request {
				pixelformat: v4l2::H264,
				size,
				sizeimage: Some(coded_size(size)),
				color: None,
			},
		)?;
		if coded.pixelformat != v4l2::H264 {
			return Err(Error::Codec(anyhow::anyhow!(
				"V4L2 encoder answered an H264 request with {}",
				v4l2::name(coded.pixelformat)
			)));
		}

		let raw = device.set_format(
			Dir::Output,
			&Request {
				pixelformat: v4l2::NV12,
				size,
				sizeimage: None,
				color: Some(config.resolved_color()),
			},
		)?;

		// Rate control spends the bitrate per unit of time, so it needs to know how
		// much time a frame is. A driver without the ioctl assumes 30.
		if let Err(err) = device.set_framerate(Dir::Output, config.framerate) {
			tracing::debug!(encoder = NAME, %err, "driver does not take a framerate");
		}

		let planes = Planes::new(&raw, size)?;

		tracing::info!(
			encoder = NAME,
			device = %device.path().display(),
			format = v4l2::name(raw.pixelformat),
			stride = raw.planes[0].stride,
			width = size.width,
			height = size.height,
			"opened H.264 encoder"
		);

		let raw = Queue::alloc(&device, Dir::Output, raw, RAW_BUFFERS)?;
		let coded = Queue::alloc(&device, Dir::Capture, coded, CODED_BUFFERS)?;

		Ok(Box::new(Self {
			device,
			raw,
			coded,
			planes,
			size,
			pending: Pending::default(),
			drainable: true,
			keyframes: true,
		}))
	}

	/// Start both queues, which has to happen with a raw frame already queued.
	///
	/// bcm2835-codec wants the first OUTPUT buffer queued before `STREAMON` and
	/// the CAPTURE queue started last. The ordering is legal on every driver, so
	/// it is unconditional rather than a quirk to detect.
	fn start(&mut self) -> Result<(), Error> {
		self.raw.stream_on(&self.device)?;
		while let Some(index) = self.coded.take_free() {
			self.coded.queue(&self.device, index, &[0], Duration::ZERO)?;
		}
		self.coded.stream_on(&self.device)
	}

	/// Take a raw buffer, waiting for the driver to release one if it holds them
	/// all.
	fn free_buffer(&mut self) -> Result<u32, Error> {
		let deadline = Instant::now() + BUFFER_TIMEOUT;
		loop {
			self.reclaim()?;
			if let Some(index) = self.raw.take_free() {
				return Ok(index);
			}
			if Instant::now() >= deadline {
				return Err(Error::Codec(anyhow::anyhow!(
					"V4L2 encoder held every input buffer for {BUFFER_TIMEOUT:?}"
				)));
			}
			self.device.wait(POLL_INTERVAL);
		}
	}

	/// Take back every raw buffer the driver has finished reading.
	///
	/// A buffer flagged `V4L2_BUF_FLAG_ERROR` is one the driver gave up on, so no
	/// access unit will ever answer that frame.
	fn reclaim(&mut self) -> Result<(), Error> {
		while let Some(buffer) = self.raw.dequeue(&self.device)?.buffer() {
			if buffer.failed() {
				tracing::warn!(encoder = NAME, buffer = buffer.index, "V4L2 encoder dropped a frame");
				self.pending.dropped(buffer.timestamp);
			}
			self.raw.reclaim(buffer.index);
		}
		Ok(())
	}

	/// Collect every access unit the driver has finished, re-queueing each coded
	/// buffer as it is emptied.
	///
	/// Returns whether the driver marked the end of a sequence, which is how a
	/// drain knows it is over.
	fn drain(&mut self, out: &mut Vec<Encoded>) -> Result<bool, Error> {
		self.reclaim()?;

		loop {
			let buffer = match self.coded.dequeue(&self.device)? {
				Dequeue::Buffer(buffer) => buffer,
				Dequeue::Empty => return Ok(false),
				// Past the buffer flagged `V4L2_BUF_FLAG_LAST`, which is the end of a
				// sequence just as much as the flag itself is.
				Dequeue::Ended => return Ok(true),
			};

			let payload = access_unit(self.coded.plane(buffer.index, 0), buffer.bytesused[0]);
			// Back to the driver before anything can fail: a coded buffer left out
			// of the pool is one the encoder never gets to write again.
			self.coded.queue(&self.device, buffer.index, &[0], Duration::ZERO)?;

			if buffer.failed() {
				// Whatever is in the buffer is not a whole access unit, which is what a
				// `coded_size` too small for a keyframe produces. Publishing it would
				// put a truncated NAL in the middle of the track.
				tracing::warn!(
					encoder = NAME,
					buffer = buffer.index,
					bytes = buffer.bytesused[0],
					"V4L2 encoder flagged an access unit bad"
				);
				self.pending.dropped(buffer.timestamp);
			} else if !payload.is_empty()
				&& let Some(payload) = self.pending.matched(buffer.timestamp, payload)
			{
				// The driver copies the raw buffer's timestamp onto the coded buffer its
				// work came out on, so this is the time of the picture that was encoded
				// rather than of whatever went in last.
				let timestamp = Timestamp::from_micros(buffer.timestamp.as_micros() as u64)?;
				out.push(Encoded::new(payload, timestamp));
			}

			if buffer.last() {
				return Ok(true);
			}
		}
	}

	/// Empty the codec's pipeline, leaving it ready for the frames that follow.
	///
	/// The kernel's drain sequence, which is the only thing that makes an encoder
	/// release a frame it is still holding: `V4L2_ENC_CMD_STOP`, dequeue CAPTURE
	/// until the buffer flagged `V4L2_BUF_FLAG_LAST`, then `V4L2_ENC_CMD_START` to
	/// resume with all the state from before the drain. See
	/// `Documentation/userspace-api/media/v4l/dev-encoder.rst`, "Drain".
	///
	/// Waiting is not a substitute for it. An encoder deeper than one-in-one-out
	/// (a lookahead, two-pass rate control) holds those frames until it is told
	/// the stream stopped, and a flush falls on every group boundary, so the first
	/// boundary would time out and take the broadcast with it. bcm2835-codec
	/// happens to be one-in-one-out, so a Pi would not have shown this.
	fn drain_tail(&mut self) -> Result<Vec<Encoded>, Error> {
		// The sequence needs both queues streaming. `VIDIOC_ENCODER_CMD` succeeds
		// without starting one otherwise, and there is nothing to drain before the
		// first frame anyway.
		if !self.raw.streaming() || !self.coded.streaming() {
			return Ok(Vec::new());
		}

		if self.drainable
			&& let Err(err) = self.device.encoder_cmd(V4L2_ENC_CMD_STOP)
		{
			// A driver with no encoder command cannot be asked to stop, so the most
			// that can be done is to take what it has already finished: complete on a
			// one-in-one-out encoder, short of the tail on anything deeper.
			tracing::warn!(
				encoder = NAME,
				%err,
				"driver takes no encoder command; a group boundary can only drain what it has finished"
			);
			self.drainable = false;
		}

		match self.drainable {
			true => self.drain_to_last(),
			false => self.wait_out(),
		}
	}

	/// Run the rest of the drain sequence, up to and including the restart.
	fn drain_to_last(&mut self) -> Result<Vec<Encoded>, Error> {
		let mut out = Vec::new();
		let mut deadline = Instant::now() + FLUSH_TIMEOUT;
		loop {
			let before = out.len();
			if self.drain(&mut out)? {
				break;
			}
			// Progress earns more time, so a slow encoder finishes a long tail while
			// a wedged one still gives up after `FLUSH_TIMEOUT` of silence.
			if out.len() > before {
				deadline = Instant::now() + FLUSH_TIMEOUT;
			} else if Instant::now() >= deadline {
				return Err(Error::Codec(anyhow::anyhow!(
					"V4L2 encoder did not finish its drain within {FLUSH_TIMEOUT:?}, holding {} frame(s)",
					self.pending.len()
				)));
			}
			self.device.wait(POLL_INTERVAL);
		}

		if !self.pending.is_empty() {
			// The drain is over, so these are frames the driver took and never
			// answered. Left standing they would make the next drain, or a fallback to
			// `wait_out`, believe the codec still owes something.
			tracing::debug!(
				encoder = NAME,
				frames = self.pending.len(),
				"V4L2 encoder ended its drain still owing access units"
			);
			self.pending.forget();
		}

		// A stopped encoder accepts OUTPUT buffers but does not process them, so
		// without this the frames after the boundary would sit in the driver.
		self.device.encoder_cmd(V4L2_ENC_CMD_START)?;
		Ok(out)
	}

	/// Take what the codec has already finished, for a driver that cannot be told
	/// to stop.
	///
	/// Only a drain where the encoder holds nothing it has not been asked for,
	/// which is what an empty [`Pending`] says here.
	fn wait_out(&mut self) -> Result<Vec<Encoded>, Error> {
		let mut out = Vec::new();
		let mut deadline = Instant::now() + FLUSH_TIMEOUT;
		loop {
			let before = out.len();
			self.drain(&mut out)?;
			if self.pending.is_empty() {
				return Ok(out);
			}
			if out.len() > before {
				deadline = Instant::now() + FLUSH_TIMEOUT;
			} else if Instant::now() >= deadline {
				return Err(Error::Codec(anyhow::anyhow!(
					"V4L2 encoder held {} frame(s) for {FLUSH_TIMEOUT:?} without encoding them",
					self.pending.len()
				)));
			}
			self.device.wait(POLL_INTERVAL);
		}
	}
}

impl Backend for V4l2 {
	fn encode(&mut self, frame: &Frame, keyframe: bool) -> Result<Vec<Encoded>, Error> {
		if frame.size() != self.size {
			return Err(Error::Codec(anyhow::anyhow!(
				"V4L2 encoder opened for {} was given a {} frame",
				self.size,
				frame.size()
			)));
		}

		let i420 = frame.surface.to_i420()?;
		// Also reclaims finished input buffers, so the codec is drained even when
		// the caller never asks for output.
		let index = self.free_buffer()?;
		self.planes.write(&mut self.raw, index, &i420)?;

		if keyframe && self.keyframes {
			// A button control: the value is ignored, the write is the request. It
			// applies to the next frame queued, so it goes in immediately before.
			//
			// Best-effort, like every other optional control here. A driver that
			// answers `EINVAL` still keeps to `V4L2_CID_MPEG_VIDEO_GOP_SIZE`, so
			// keyframes land on the GOP boundary instead of where the caller asked
			// for one, which is a worse group layout rather than a broken stream. As a
			// hard error it would have failed the very first frame, since that one is
			// always requested.
			self.keyframes = self.device.try_control(V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME, 0);
			if !self.keyframes {
				tracing::warn!(
					encoder = NAME,
					device = %self.device.path().display(),
					"driver takes no keyframe request; groups fall on the encoder's own GOP boundary"
				);
			}
		}

		let timestamp = Duration::from_micros(frame.timestamp.as_micros() as u64);
		let bytesused: Vec<u32> = self.raw.format().planes.iter().map(|plane| plane.sizeimage).collect();
		self.raw.queue(&self.device, index, &bytesused, timestamp)?;
		self.pending.queued(timestamp);

		if !self.raw.streaming() {
			self.start()?;
		}

		let mut out = Vec::new();
		self.drain(&mut out)?;
		Ok(out)
	}

	fn flush(&mut self) -> Result<Vec<Encoded>, Error> {
		self.drain_tail()
	}

	fn finish(&mut self) -> Result<Vec<Encoded>, Error> {
		self.drain_tail()
	}

	fn set_bitrate(&mut self, bitrate: u64) -> Result<(), Error> {
		// Settable on a streaming encoder and applied without an IDR, which is
		// exactly what the congestion controller wants. A driver that refuses says
		// so once and is not asked again.
		set_bitrate(&self.device, bitrate).map_err(|err| {
			tracing::debug!(encoder = NAME, %err, "driver refused a bitrate change");
			Error::BitrateUnsupported(NAME)
		})
	}

	fn name(&self) -> &str {
		NAME
	}
}

/// Which frame each coded buffer answers, and what to do with one that answers
/// none.
///
/// The driver copies an OUTPUT buffer's timestamp onto the CAPTURE buffer its
/// work came out on, so the timestamp is the only thing tying an access unit
/// back to a frame. Counting the two against each other does not work: under
/// `V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE` the first coded buffer is SPS/PPS
/// alone, answering no frame, and counting it would leave the encoder reporting
/// itself drained one access unit early at every group boundary while the
/// straggler surfaced in the next group ahead of its keyframe.
#[derive(Debug, Default)]
struct Pending {
	/// Timestamps queued and not yet answered, in the order they were queued.
	frames: VecDeque<Duration>,
	/// Bytes from a coded buffer that answered no frame, waiting for the access
	/// unit to go in front of.
	header: Option<Bytes>,
}

impl Pending {
	/// Record a frame handed to the driver.
	fn queued(&mut self, timestamp: Duration) {
		self.frames.push_back(timestamp);
	}

	/// How many frames the driver has taken and not answered.
	fn len(&self) -> usize {
		self.frames.len()
	}

	fn is_empty(&self) -> bool {
		self.frames.is_empty()
	}

	/// The access unit a coded buffer holds, or `None` when it answers no frame.
	///
	/// Parameter sets that arrived on their own go in front of the next access
	/// unit instead of being published as a frame of their own, which is both what
	/// `HEADER_MODE_JOINED_WITH_1ST_FRAME` would have produced and what a
	/// subscriber joining at that keyframe needs.
	fn matched(&mut self, timestamp: Duration, payload: Bytes) -> Option<Bytes> {
		let Some(at) = self.frames.iter().position(|frame| *frame == timestamp) else {
			self.header = Some(match self.header.take() {
				Some(header) => join(&header, &payload),
				None => payload,
			});
			return None;
		};

		// Frames ahead of the match were taken and never answered, and an encoder
		// does not go back: nothing will answer them now.
		self.frames.drain(..=at);
		Some(match self.header.take() {
			Some(header) => join(&header, &payload),
			None => payload,
		})
	}

	/// Forget the frame a buffer answered without a usable access unit.
	fn dropped(&mut self, timestamp: Duration) {
		if let Some(at) = self.frames.iter().position(|frame| *frame == timestamp) {
			self.frames.drain(..=at);
		}
	}

	/// Forget every outstanding frame, for a drain the driver has declared over.
	fn forget(&mut self) {
		self.frames.clear();
	}
}

/// Concatenate two pieces of one access unit, which are already Annex-B and so
/// need nothing between them.
fn join(header: &[u8], payload: &[u8]) -> Bytes {
	let mut joined = BytesMut::with_capacity(header.len() + payload.len());
	joined.extend_from_slice(header);
	joined.extend_from_slice(payload);
	joined.freeze()
}

fn set_bitrate(device: &Device, bitrate: u64) -> Result<(), Error> {
	device.set_control(V4L2_CID_MPEG_VIDEO_BITRATE, bitrate.min(i32::MAX as u64) as i32)
}

/// The H.264 level a resolution needs, as the `V4L2_CID_MPEG_VIDEO_H264_LEVEL`
/// menu spells it.
///
/// Chosen on frame size in macroblocks, which is the `MaxFS` column of Table A-1
/// and the constraint that actually bites: bcm2835-codec defaults to level 1.0,
/// whose 396-macroblock limit is 128x96. The other level constraints (bit rate,
/// decoded picture buffer) are looser than what this hardware does anyway.
fn h264_level(size: Size) -> i32 {
	let macroblocks = size.width.div_ceil(16) * size.height.div_ceil(16);
	let level = match macroblocks {
		// 352x288
		0..=396 => v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_1_3,
		// 720x576
		397..=1620 => v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_3_0,
		// 1280x720
		1621..=3600 => v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_3_1,
		// 1920x1088
		3601..=8192 => v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_4_0,
		_ => v4l2_mpeg_video_h264_level_V4L2_MPEG_VIDEO_H264_LEVEL_5_1,
	};
	level as i32
}

/// How large a coded buffer to ask for, since the driver cannot size an access
/// unit from the picture dimensions.
///
/// Half a raw frame is well clear of what a keyframe costs at any sane quality,
/// and the floor keeps a small picture's buffer larger than its own first
/// keyframe.
fn coded_size(size: Size) -> u32 {
	const MIN: u32 = 256 * 1024;
	const MAX: u32 = 4 * 1024 * 1024;
	(size.pixels().min(u32::MAX as u64) as u32 / 2).clamp(MIN, MAX)
}

/// Copy an access unit out of a coded buffer, without the driver's padding.
///
/// Trailing zeroes past the last NAL are `trailing_zero_8bits`, which a decoder
/// discards, and some drivers pad a coded buffer with them rather than reporting
/// the exact length. Dropping them is only safe because the profile is
/// constrained baseline, hence CAVLC: a CABAC stream can carry meaningful
/// `cabac_zero_words` in the same position.
fn access_unit(buffer: &[u8], bytesused: u32) -> Bytes {
	let used = (bytesused as usize).min(buffer.len());
	let end = buffer[..used]
		.iter()
		.rposition(|byte| *byte != 0)
		.map_or(0, |at| at + 1);
	Bytes::copy_from_slice(&buffer[..end])
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The level has to clear the resolution, or bcm2835-codec refuses the format
	/// outright. Spot-checked against Table A-1 rather than the driver's menu, so
	/// a driver with a different menu ordering still gets a correct level.
	#[test]
	fn the_level_clears_the_resolution() {
		assert_eq!(h264_level(Size::new(320, 240)), 4); // 1.3
		assert_eq!(h264_level(Size::new(640, 480)), 8); // 3.0
		assert_eq!(h264_level(Size::new(1280, 720)), 9); // 3.1
		assert_eq!(h264_level(Size::new(1920, 1080)), 11); // 4.0
		assert_eq!(h264_level(Size::new(3840, 2160)), 15); // 5.1
	}

	#[test]
	fn the_coded_buffer_is_bounded() {
		assert_eq!(coded_size(Size::new(160, 120)), 256 * 1024);
		assert_eq!(coded_size(Size::new(1920, 1080)), 1920 * 1080 / 2);
		assert_eq!(coded_size(Size::new(7680, 4320)), 4 * 1024 * 1024);
	}

	/// A driver that pipelines answers a frame several buffers later, and only
	/// the timestamp says which frame that was.
	#[test]
	fn an_access_unit_is_matched_to_the_frame_it_came_from() {
		let mut pending = Pending::default();
		for micros in [0, 33_000, 66_000] {
			pending.queued(Duration::from_micros(micros));
		}
		assert_eq!(pending.len(), 3);

		let payload = Bytes::from_static(&[1, 2, 3]);
		assert_eq!(
			pending.matched(Duration::from_micros(0), payload.clone()),
			Some(payload.clone())
		);
		assert_eq!(pending.len(), 2);

		// A frame the driver skipped goes with the one that overtook it, since an
		// encoder never comes back to it.
		assert!(pending.matched(Duration::from_micros(66_000), payload).is_some());
		assert!(pending.is_empty());
	}

	/// The case a bare count gets wrong: a coded buffer that answers no frame at
	/// all, which is what `HEADER_MODE_SEPARATE` produces.
	#[test]
	fn separate_parameter_sets_join_the_next_access_unit() {
		let mut pending = Pending::default();
		pending.queued(Duration::from_micros(500));

		// SPS/PPS on their own, under whatever timestamp the driver left behind.
		assert_eq!(
			pending.matched(Duration::from_micros(0), Bytes::from_static(&[0, 0, 0, 1, 0x67])),
			None
		);
		// Still owed the frame, which is what stops a flush finishing early.
		assert_eq!(pending.len(), 1);

		let joined = pending
			.matched(Duration::from_micros(500), Bytes::from_static(&[0, 0, 0, 1, 0x65]))
			.unwrap();
		assert_eq!(&joined[..], &[0, 0, 0, 1, 0x67, 0, 0, 0, 1, 0x65]);
		assert!(pending.is_empty());
		// Carried once, not onto every access unit after it.
		let next = Bytes::from_static(&[0, 0, 0, 1, 0x41]);
		pending.queued(Duration::from_micros(533));
		assert_eq!(pending.matched(Duration::from_micros(533), next.clone()), Some(next));
	}

	/// A frame the driver flagged bad is owed by nobody, and forgetting it must
	/// not shift the frames queued after it.
	#[test]
	fn a_dropped_frame_leaves_the_rest_matchable() {
		let mut pending = Pending::default();
		for micros in [10, 20, 30] {
			pending.queued(Duration::from_micros(micros));
		}

		pending.dropped(Duration::from_micros(10));
		assert_eq!(pending.len(), 2);
		// A timestamp that was never queued changes nothing.
		pending.dropped(Duration::from_micros(999));
		assert_eq!(pending.len(), 2);

		let payload = Bytes::from_static(&[9]);
		assert_eq!(
			pending.matched(Duration::from_micros(20), payload.clone()),
			Some(payload)
		);
		assert_eq!(pending.len(), 1);
	}

	/// The payload stops at the last non-zero byte, and a buffer the driver wrote
	/// nothing into is not an access unit at all.
	#[test]
	fn padding_is_not_part_of_the_access_unit() {
		let buffer = [0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 0];
		assert_eq!(&access_unit(&buffer, 10)[..], &[0, 0, 0, 1, 0x65, 0x88]);
		assert!(access_unit(&[0; 8], 8).is_empty());
		// `bytesused` past the mapping is clamped rather than trusted.
		assert_eq!(access_unit(&buffer, 64).len(), 6);
	}
}
