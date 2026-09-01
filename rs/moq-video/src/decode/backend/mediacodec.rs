//! Hardware H.264 / H.265 / AV1 decode backend via Android MediaCodec
//! (`AMediaCodec`).
//!
//! The inverse of the encode MediaCodec backend, and the one place the two
//! differ in shape: the decoder is configured with an `ImageReader`'s surface as
//! its output rather than with CPU output buffers. A decoded picture is then an
//! `AHardwareBuffer` a GL or Vulkan consumer imports directly, which is what
//! [`Surface::HardwareBuffer`] carries, and the read-back to I420 takes the row
//! and pixel strides off the image instead of guessing at the device's padding.
//! The ByteBuffer output path offers neither.
//!
//! Access units arrive Annex-B with the parameter sets inline ahead of each
//! keyframe (the front end converts avc1 / hvc1 for us), which MediaCodec takes
//! as-is, so no `csd-0` / `csd-1` configuration is needed and nothing here parses
//! an avcC record.
//!
//! Two queues run in step. The codec is fed access units and hands back output
//! buffers, each carrying its own presentation time, since a decoder emits
//! pictures in display order while it is fed in decode order. Releasing one of
//! those buffers for rendering queues the picture into the reader, where it is
//! acquired as an image; that hop carries no timestamp of its own, so the
//! release order is recorded and each acquired image is paired with it.
//!
//! Frames hold slots in the reader's queue, so a consumer that hoards them
//! stalls decoding. That is the same trade the VideoToolbox and NVDEC backends
//! make: draw and drop.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use moq_net::Timestamp;
use ndk::hardware_buffer::HardwareBufferUsage;
use ndk::media::image_reader::{AcquireResult, Image, ImageFormat, ImageReader};
use ndk::media::media_codec::{self, DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodecDirection};
use ndk::media::media_format::MediaFormat;
use ndk::media_error::MediaError;

use super::{Backend, Codec, Config};
use crate::frame::{
	Surface,
	android::{HardwareBuffer, Reader},
};
use crate::{Error, Frame};

pub(crate) const NAME: &str = "mediacodec";

/// The MIME types MediaCodec names the codecs by.
const MIME_H264: &str = "video/avc";
const MIME_H265: &str = "video/hevc";
const MIME_AV1: &str = "video/av01";

// `AMediaFormat` keys. The NDK ships them as `AMEDIAFORMAT_KEY_*` string
// constants that the `ndk` crate does not re-export; they are the same strings
// `android.media.MediaFormat` documents.
const KEY_MIME: &str = "mime";
const KEY_WIDTH: &str = "width";
const KEY_HEIGHT: &str = "height";
const KEY_LOW_LATENCY: &str = "low-latency";
const KEY_PRIORITY: &str = "priority";

/// `PRIORITY_REALTIME`: this is a live stream, not a file being played back.
const PRIORITY_REALTIME: i32 = 0;

/// `AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG`, likewise not re-exported by the `ndk`
/// crate.
const FLAG_CODEC_CONFIG: u32 = 2;

/// The size the reader's queue is created at.
///
/// Only a default: MediaCodec sets the real geometry on the window once it has
/// parsed the stream, and each image reports what it actually got, which is what
/// a frame's size comes from. Sized for the common case so the queue usually
/// does not have to reallocate.
const DEFAULT_SIZE: (i32, i32) = (1920, 1080);

/// How many decoded pictures the reader's queue holds.
///
/// It has to cover the decoder's reference pictures (4 or so for H.264), the
/// consumer's playout buffer, and whatever is in flight between them; anything
/// less and the decoder stalls waiting for a slot that a consumer holding two
/// frames could have freed.
const QUEUE_DEPTH: i32 = 8;

/// How long to wait for a free input buffer on each round.
const INPUT_TIMEOUT: Duration = Duration::from_millis(10);

/// Poll for decoded pictures without waiting. [`MediaCodec::decode`] drains on
/// both sides of every submission, so a picture that is not ready yet is
/// collected on the next access unit rather than waited for here.
const OUTPUT_TIMEOUT: Duration = Duration::ZERO;

/// How many rounds a submission spends freeing an input buffer before giving up.
const SUBMIT_ROUNDS: u32 = 20;

pub(crate) struct MediaCodec {
	/// Declared first so it is dropped first: the codec writes into the reader's
	/// queue, so it has to stop before the reader can go.
	codec: media_codec::MediaCodec,
	/// Shared with every frame handed out, since deleting the reader invalidates
	/// images already acquired from it.
	reader: Arc<Reader>,
	/// Timestamps of the pictures released for rendering whose image has not been
	/// acquired yet, oldest first. The reader hands images back in the order they
	/// were queued but carries no timestamp of its own, so this is what pairs the
	/// two up.
	/// Bounded by the reader's queue depth: a timestamp is pushed when a picture
	/// is released for rendering and popped when its image is acquired, and the
	/// codec cannot release more than [`QUEUE_DEPTH`] pictures without one being
	/// acquired first.
	rendered: VecDeque<Timestamp>,
	/// Whether the last acquire found every slot held, so the stall is reported
	/// once on the way in rather than on every poll.
	stalled: bool,
}

// SAFETY: `AMediaCodec` and `AImageReader` are owned handles with no thread
// affinity; the NDK only requires that calls on one of them are serialized,
// which they are because every method here takes `&mut self` and one decode task
// owns the backend outright. `Send` is what lets the boxed trait object satisfy
// `Backend: Send`.
unsafe impl Send for MediaCodec {}

impl MediaCodec {
	/// Open a decoder for `codec`.
	///
	/// `config` is accepted for signature parity: MediaCodec decodes at the
	/// stream's native size and has no scaler to point
	/// [`Config::resize`](crate::decode::Config) at.
	pub(crate) fn open(codec: Codec, _config: &Config) -> Result<Box<dyn Backend>, Error> {
		let mime = match codec {
			Codec::H264 => MIME_H264,
			Codec::H265 => MIME_H265,
			Codec::Av1 => MIME_AV1,
		};

		// `GPU_SAMPLED_IMAGE` is what a consumer importing the buffer as a texture
		// needs; `CPU_READ_OFTEN` is what makes the read-back in
		// `Surface::into_i420` work at all, at the cost of a layout the device may
		// otherwise have tiled.
		let (width, height) = DEFAULT_SIZE;
		let reader = ImageReader::new_with_usage(
			width,
			height,
			ImageFormat::YUV_420_888,
			HardwareBufferUsage::GPU_SAMPLED_IMAGE | HardwareBufferUsage::CPU_READ_OFTEN,
			QUEUE_DEPTH,
		)
		.map_err(|e| reader_err("create an ImageReader", e))?;
		let surface = reader.window().map_err(|e| reader_err("get the reader's window", e))?;

		let decoder = media_codec::MediaCodec::from_decoder_type(mime)
			.ok_or_else(|| Error::Codec(anyhow::anyhow!("no MediaCodec decoder for {mime}")))?;
		let format = decoder_format(mime, width, height);
		decoder
			.configure(&format, Some(&surface), MediaCodecDirection::Decoder)
			.map_err(|e| codec_err("configure", e))?;
		decoder.start().map_err(|e| codec_err("start", e))?;

		tracing::info!(decoder = NAME, codec = codec.label(), "opened video decoder");
		Ok(Box::new(Self {
			codec: decoder,
			reader: Arc::new(Reader::new(reader)),
			rendered: VecDeque::new(),
			stalled: false,
		}))
	}

	/// Copy one access unit into a free input buffer.
	///
	/// Waits for a buffer rather than dropping the access unit when the codec has
	/// none free: a decoder that skips one produces a broken picture until the
	/// next keyframe, which is a far worse trade than the wait.
	fn submit(&mut self, access_unit: &[u8], timestamp: Timestamp, out: &mut Vec<Frame>) -> Result<(), Error> {
		let time = timestamp.as_micros().min(u64::MAX as u128) as u64;

		for _ in 0..SUBMIT_ROUNDS {
			let queued = match self
				.codec
				.dequeue_input_buffer(INPUT_TIMEOUT)
				.map_err(|e| codec_err("dequeue an input buffer", e))?
			{
				DequeuedInputBufferResult::Buffer(mut buffer) => {
					let target = buffer.buffer_mut();
					if target.len() < access_unit.len() {
						return Err(Error::Codec(anyhow::anyhow!(
							"MediaCodec input buffer is {} bytes, needs {} for this access unit",
							target.len(),
							access_unit.len()
						)));
					}
					// SAFETY: `target` is at least `access_unit.len()` bytes (checked
					// just above), the two are distinct allocations (a codec input buffer
					// and the caller's payload), and `MaybeUninit<u8>` shares the layout
					// of `u8`, so copying initialized bytes over it is what initializes
					// it.
					unsafe {
						std::ptr::copy_nonoverlapping(
							access_unit.as_ptr(),
							target.as_mut_ptr().cast::<u8>(),
							access_unit.len(),
						);
					}
					self.codec
						.queue_input_buffer(buffer, 0, access_unit.len(), time, 0)
						.map_err(|e| codec_err("queue an input buffer", e))?;
					true
				}
				DequeuedInputBufferResult::TryAgainLater => false,
			};
			if queued {
				return Ok(());
			}

			// Every input buffer is still with the codec; collecting its output is
			// what frees one.
			self.drain(OUTPUT_TIMEOUT, out)?;
		}

		Err(Error::Codec(anyhow::anyhow!(
			"MediaCodec never freed an input buffer for an access unit"
		)))
	}

	/// Render every decoded picture the codec has ready and collect the images
	/// they turn into.
	fn drain(&mut self, timeout: Duration, out: &mut Vec<Frame>) -> Result<(), Error> {
		loop {
			let rendered = match self
				.codec
				.dequeue_output_buffer(timeout)
				.map_err(|e| codec_err("dequeue an output buffer", e))?
			{
				DequeuedOutputBufferInfoResult::Buffer(buffer) => {
					let info = *buffer.info();
					// Never `OutputBuffer::buffer`: a codec configured with a surface has
					// no CPU-visible output buffer, and asking for one panics. An empty
					// buffer is the codec's own marker rather than a picture, and a
					// codec-config one is the parameter sets echoed back.
					let render = info.flags() & FLAG_CODEC_CONFIG == 0 && info.size() > 0;
					self.codec
						.release_output_buffer(buffer, render)
						.map_err(|e| codec_err("release an output buffer", e))?;
					// The presentation time the picture came back with, which is the one
					// it was fed in with: a decoder emits in display order, so the access
					// unit being submitted is not the one coming out.
					render.then(|| info.presentation_time_us())
				}
				DequeuedOutputBufferInfoResult::TryAgainLater => {
					self.collect(out)?;
					return Ok(());
				}
				// The stream's geometry is read off each image rather than off the
				// format, and the parameter sets are inline, so neither of these
				// changes anything here.
				DequeuedOutputBufferInfoResult::OutputFormatChanged
				| DequeuedOutputBufferInfoResult::OutputBuffersChanged => None,
			};

			if let Some(micros) = rendered {
				self.rendered.push_back(Timestamp::from_micros(micros.max(0) as u64)?);
			}
			// Eagerly, because releasing for rendering queues the picture into the
			// reader before it returns, so the image is normally there already and
			// waiting for the next access unit would add a frame of latency.
			self.collect(out)?;
		}
	}

	/// Acquire every rendered picture the reader has ready, pairing each with the
	/// timestamp of the buffer that produced it.
	fn collect(&mut self, out: &mut Vec<Frame>) -> Result<(), Error> {
		while let Some(&timestamp) = self.rendered.front() {
			match self
				.reader
				.acquire_next_image()
				.map_err(|e| reader_err("acquire an image", e))?
			{
				AcquireResult::Image(image) => {
					self.rendered.pop_front();
					let (width, height) = image_size(&image)?;
					let buffer = HardwareBuffer::new(self.reader.clone(), image, width, height);
					out.push(Frame::new(Surface::HardwareBuffer(buffer), timestamp));
				}
				// The picture is on its way but not queued yet, so it comes out on the
				// next round rather than being lost.
				AcquireResult::NoBufferAvailable => {
					self.stalled = false;
					break;
				}
				// Every slot in the queue is held by a consumer, so there is nowhere
				// for this picture to go until one is dropped. Decoding is stalled
				// behind whoever is hoarding frames, which is worth saying out loud.
				AcquireResult::MaxImagesAcquired => {
					// Once, on the way into the stall. `collect` runs two or three
					// times per access unit, so warning on every poll would log at
					// frame rate for as long as the consumer holds its frames.
					if !self.stalled {
						self.stalled = true;
						tracing::warn!(
							decoder = NAME,
							depth = QUEUE_DEPTH,
							"every decoded frame is still held by a consumer; decoding is stalled"
						);
					}
					break;
				}
			}
		}
		Ok(())
	}
}

impl Backend for MediaCodec {
	fn decode(&mut self, access_unit: Bytes, timestamp: Timestamp, _keyframe: bool) -> Result<Vec<Frame>, Error> {
		let mut out = Vec::new();
		// Collect what the codec finished while the caller was elsewhere, which is
		// also what frees the input buffer the submission below needs.
		self.drain(OUTPUT_TIMEOUT, &mut out)?;
		self.submit(&access_unit, timestamp, &mut out)?;
		self.drain(OUTPUT_TIMEOUT, &mut out)?;
		Ok(out)
	}

	fn name(&self) -> &str {
		NAME
	}
}

/// The `AMediaFormat` describing the stream to decode.
///
/// The dimensions are a starting guess that the first keyframe's parameter sets
/// correct, since nothing upstream of a backend knows the coded size. The
/// parameter sets ride the bitstream, so there is no `csd-0` / `csd-1` to set.
fn decoder_format(mime: &str, width: i32, height: i32) -> MediaFormat {
	let mut format = MediaFormat::new();
	format.set_str(KEY_MIME, mime);
	format.set_i32(KEY_WIDTH, width);
	format.set_i32(KEY_HEIGHT, height);
	// Ask the decoder to hold as little as it can, which is the difference
	// between a live stream and a file. A hint an older device drops.
	format.set_i32(KEY_LOW_LATENCY, 1);
	format.set_i32(KEY_PRIORITY, PRIORITY_REALTIME);
	format
}

/// The picture's visible size, both dimensions rounded down to even.
///
/// The crop rectangle rather than the buffer dimensions: a coded picture is
/// padded up to a macroblock multiple, so 1080 lines arrive as a 1088-line
/// buffer, and the crop is what says how much of it is the picture. Taken only
/// when the crop is anchored at the origin, since the read-back walks each plane
/// from there; a decoder that offsets its crop gets the whole buffer rather than
/// a picture shifted by the offset.
fn image_size(image: &Image) -> Result<(u32, u32), Error> {
	let width = image.width().map_err(|e| reader_err("read an image's width", e))?;
	let height = image.height().map_err(|e| reader_err("read an image's height", e))?;
	let crop = image.crop_rect().map_err(|e| reader_err("read an image's crop", e))?;

	let (mut w, mut h) = (width, height);
	let (cropped_w, cropped_h) = (crop.right - crop.left, crop.bottom - crop.top);
	if crop.left == 0 && crop.top == 0 && cropped_w > 0 && cropped_h > 0 && cropped_w <= w && cropped_h <= h {
		(w, h) = (cropped_w, cropped_h);
	}

	// 4:2:0 chroma is 2x2, so an odd dimension has no whole chroma sample to go
	// with its last row or column.
	let (w, h) = (w.max(0) as u32 & !1, h.max(0) as u32 & !1);
	if w == 0 || h == 0 {
		return Err(Error::Codec(anyhow::anyhow!(
			"MediaCodec produced a {width}x{height} image, which is not a picture"
		)));
	}
	Ok((w, h))
}

/// Wrap an NDK media error from the codec, naming the call that produced it.
fn codec_err(what: &str, error: MediaError) -> Error {
	Error::Codec(anyhow::anyhow!("failed to {what} on the MediaCodec decoder: {error}"))
}

/// Wrap an NDK media error from the reader, naming the call that produced it.
fn reader_err(what: &str, error: MediaError) -> Error {
	Error::Codec(anyhow::anyhow!(
		"failed to {what} on the decoder's ImageReader: {error}"
	))
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The hardware round trip: encode a picture with the MediaCodec encoder and
	/// decode it back, which is the only way to see that the surface output path
	/// produces a frame at all.
	#[test]
	#[ignore = "needs an Android device with MediaCodec H.264 hardware"]
	fn decodes_what_the_encoder_produced() {
		use crate::encode::{Config as EncodeConfig, Kind as EncodeKind};

		let size = crate::Size::new(320, 240);
		let mut config = EncodeConfig::new(size.width, size.height, 30);
		config.kind = EncodeKind::Named(NAME.to_owned());
		let mut encoder = crate::encode::Encoder::new(&config).expect("a MediaCodec encoder");

		let i420 = crate::I420::new(
			size.width,
			size.height,
			vec![0x80; crate::I420::len(size.width, size.height)],
		)
		.unwrap();

		let mut decoder = MediaCodec::open(Codec::H264, &Config::new()).expect("a MediaCodec decoder");
		let mut frames = Vec::new();
		for index in 0..30u64 {
			let timestamp = Timestamp::from_micros(index * 33_333).unwrap();
			let frame = Frame::new(Surface::I420(i420.clone()), timestamp);
			encoder.keyframe();
			for encoded in encoder.encode(&frame).unwrap() {
				frames.extend(decoder.decode(encoded.payload, encoded.timestamp, true).unwrap());
			}
		}

		let frame = frames.first().expect("at least one decoded frame");
		assert_eq!(frame.size(), size);
		assert!(
			matches!(frame.surface, Surface::HardwareBuffer(_)),
			"a decoded picture should stay in its hardware buffer",
		);

		// And it has to survive the read-back, which is the arm every CPU
		// consumer takes and the only exercise `download_i420`'s stride walking
		// gets. Mid-gray in, mid-gray out.
		let frames = frames.into_iter().next().expect("checked above");
		let i420 = frames.surface.into_i420().expect("read back to I420");
		assert_eq!(i420.len(), crate::I420::len(size.width, size.height));
		let luma = &i420[..(size.width * size.height) as usize];
		assert!(
			luma.iter().all(|byte| byte.abs_diff(0x80) <= 8),
			"the read-back luma plane should still be mid-gray",
		);
	}
}
