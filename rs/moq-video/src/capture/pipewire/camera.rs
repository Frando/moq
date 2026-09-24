//! PipeWire camera capture: the `Video/Source` nodes with the `Camera` role,
//! which is how PipeWire presents V4L2 webcams (spa-v4l2) and libcamera sensors
//! such as a Raspberry Pi CSI camera (spa-libcamera). A CSI camera has no V4L2
//! node that yields processed frames, so on a Pi this is the only way to open it
//! as a camera.
//!
//! Outside a sandbox this connects to the session's PipeWire socket, which also
//! works headless. A sandbox cannot reach that socket, so there the camera portal
//! (`org.freedesktop.portal.Camera`) grants access and hands back a PipeWire
//! remote that exposes the cameras. Streaming reuses the parent module's loop.

use std::cell::RefCell;
use std::os::fd::OwnedFd;
use std::rc::Rc;
use std::time::Duration;

use pipewire as pw;

use super::{Capture, Kind, Target, err};
use crate::Error;
use crate::capture::{Camera, Config, PIPEWIRE, Stream};

/// How long the PipeWire daemon may take to list its objects. A connected
/// daemon answers in milliseconds, so running out means it is stuck.
const SCAN_TIMEOUT: Duration = Duration::from_secs(5);

/// List the PipeWire cameras as [`Camera`]s with `pipewire:<node name>` ids.
pub(in crate::capture) async fn cameras() -> Result<Vec<Camera>, Error> {
	let remote = remote().await?;
	let sandboxed = remote.is_some();
	let nodes = crate::capture::blocking(move || {
		pw::init();
		let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| err("pipewire main loop", e))?;
		let context = pw::context::ContextRc::new(&mainloop, None).map_err(|e| err("pipewire context", e))?;
		let core = match super::connect(&context, remote) {
			Ok(core) => core,
			// PipeWire is optional on the host, and a host without it has no
			// PipeWire cameras. A portal remote that fails to connect is broken.
			Err(error) if !sandboxed => {
				tracing::debug!(%error, "no PipeWire session to list cameras from");
				return Ok(Vec::new());
			}
			Err(error) => return Err(err("pipewire connect", error)),
		};
		scan(&mainloop, &core)
	})
	.await?;

	Ok(nodes
		.into_iter()
		.map(|node| Camera {
			id: format!("{PIPEWIRE}:{}", node.name),
			name: node.description,
		})
		.collect())
}

/// Open the camera node named `node`, or the session manager's default camera.
///
/// The camera streams its first raw mode that this backend converts, so
/// `config.width` and `config.height` are not applied yet. The parent module's
/// `Capture::size` says why the offers carry no size.
pub(in crate::capture) async fn open(config: &Config, node: Option<&str>) -> Result<Stream, Error> {
	let remote = remote().await?;
	let label = match node {
		Some(node) => format!("{PIPEWIRE}:{node}"),
		None => PIPEWIRE.to_string(),
	};
	super::start(
		config,
		Capture {
			kind: Kind::Camera,
			remote,
			target: Target::Camera(node.map(str::to_string)),
			label,
			size: None,
		},
		None,
	)
	.await
}

/// The camera portal's PipeWire remote inside a sandbox, or `None` to use the
/// session's socket.
async fn remote() -> Result<Option<OwnedFd>, Error> {
	if !ashpd::is_sandboxed() {
		return Ok(None);
	}
	let portal = ashpd::desktop::camera::Camera::new()
		.await
		.map_err(|e| err("camera portal", e))?;
	if !portal.is_present().await.map_err(|e| err("camera portal", e))? {
		return Err(Error::SourceUnavailable(
			"the camera portal reports no camera".to_string(),
		));
	}
	// The portal asks the user unless the sandbox's permission store already
	// holds a grant, so this blocks on the user the first time.
	portal
		.request_access(Default::default())
		.await
		.map_err(|e| err("camera portal access", e))?
		.response()
		.map_err(|e| Error::PermissionDenied(format!("camera access request: {e}")))?;
	let fd = portal
		.open_pipe_wire_remote(Default::default())
		.await
		.map_err(|e| err("camera portal pipewire remote", e))?;
	Ok(Some(fd))
}

/// A PipeWire camera node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Node {
	/// The registry id to link to. Only valid for this connection's lifetime.
	id: u32,
	/// `node.name`, which stays the same across restarts and reboots.
	name: String,
	/// `node.description`, or the best name the node has short of it.
	description: String,
}

impl Node {
	/// Build a node from a registry global's properties, or `None` when it is
	/// not a camera. The criterion is the one xdg-desktop-portal applies when it
	/// decides which nodes a sandboxed app may see.
	fn from_props<'a>(id: u32, prop: impl Fn(&str) -> Option<&'a str>) -> Option<Self> {
		if prop(*pw::keys::MEDIA_CLASS)? != "Video/Source" || prop(*pw::keys::MEDIA_ROLE)? != "Camera" {
			return None;
		}
		let name = prop(*pw::keys::NODE_NAME).filter(|name| !name.is_empty())?;
		let description = prop(*pw::keys::NODE_DESCRIPTION)
			.or_else(|| prop(*pw::keys::NODE_NICK))
			.filter(|description| !description.is_empty())
			.unwrap_or(name);
		Some(Self {
			id,
			name: name.to_string(),
			description: description.to_string(),
		})
	}
}

/// List the camera nodes on `core`, running `mainloop` until the daemon has
/// sent every registry global.
pub(super) fn scan(mainloop: &pw::main_loop::MainLoopRc, core: &pw::core::CoreRc) -> Result<Vec<Node>, Error> {
	let registry = core.get_registry_rc().map_err(|e| err("pipewire registry", e))?;
	let nodes = Rc::new(RefCell::new(Vec::new()));
	let outcome: Rc<RefCell<Option<Result<(), Error>>>> = Rc::new(RefCell::new(None));

	// The daemon answers a sync only after every global it announced before it.
	let pending = core.sync(0).map_err(|e| err("pipewire sync", e))?;
	let _core = core
		.add_listener_local()
		.done({
			let outcome = outcome.clone();
			let mainloop = mainloop.downgrade();
			move |id, seq| {
				if id == pw::core::PW_ID_CORE && seq == pending {
					outcome.borrow_mut().get_or_insert(Ok(()));
					if let Some(mainloop) = mainloop.upgrade() {
						mainloop.quit();
					}
				}
			}
		})
		.error({
			let outcome = outcome.clone();
			let mainloop = mainloop.downgrade();
			move |id, _, _, message| {
				if id == pw::core::PW_ID_CORE {
					outcome.borrow_mut().get_or_insert(Err(err("pipewire", message)));
					if let Some(mainloop) = mainloop.upgrade() {
						mainloop.quit();
					}
				}
			}
		})
		.register();
	let _registry = registry
		.add_listener_local()
		.global({
			let nodes = nodes.clone();
			move |global| {
				if global.type_ != pw::types::ObjectType::Node {
					return;
				}
				let Some(props) = global.props else { return };
				if let Some(node) = Node::from_props(global.id, |key| props.get(key)) {
					nodes.borrow_mut().push(node);
				}
			}
		})
		.register();
	let timer = mainloop.loop_().add_timer({
		let mainloop = mainloop.downgrade();
		move |_| {
			if let Some(mainloop) = mainloop.upgrade() {
				mainloop.quit();
			}
		}
	});
	timer
		.update_timer(Some(SCAN_TIMEOUT), None)
		.into_result()
		.map_err(|e| err("pipewire timer", e))?;

	mainloop.run();
	let outcome = outcome.borrow_mut().take();
	match outcome {
		Some(Ok(())) => Ok(nodes.take()),
		Some(Err(error)) => Err(error),
		None => Err(Error::Codec(anyhow::anyhow!(
			"PipeWire did not list its objects within {SCAN_TIMEOUT:?}"
		))),
	}
}

/// The node id to link to for `name`, or `None` to let the session manager
/// pick its default camera. Fails when the camera is not there, rather than
/// leaving the stream unlinked until the format wait runs out.
pub(super) fn resolve(nodes: &[Node], name: Option<&str>) -> Result<Option<u32>, Error> {
	let Some(name) = name else {
		if nodes.is_empty() {
			return Err(Error::SourceUnavailable("PipeWire has no camera".to_string()));
		}
		return Ok(None);
	};
	match nodes.iter().find(|node| node.name == name) {
		Some(node) => Ok(Some(node.id)),
		None => {
			let found: Vec<_> = nodes.iter().map(|node| node.name.as_str()).collect();
			Err(Error::SourceUnavailable(format!(
				"no PipeWire camera named {name} (found: {})",
				found.join(", ")
			)))
		}
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use super::*;

	fn node(props: &[(&str, &str)]) -> Option<Node> {
		let props: HashMap<_, _> = props.iter().copied().collect();
		Node::from_props(62, |key| props.get(key).copied())
	}

	const WEBCAM: [(&str, &str); 4] = [
		("media.class", "Video/Source"),
		("media.role", "Camera"),
		("node.name", "v4l2_input.pci-0000_00_14.0-usb-0_4_1.0"),
		("node.description", "Integrated Camera (V4L2)"),
	];

	#[test]
	fn camera_nodes_are_video_sources_with_the_camera_role() {
		assert_eq!(
			node(&WEBCAM),
			Some(Node {
				id: 62,
				name: "v4l2_input.pci-0000_00_14.0-usb-0_4_1.0".to_string(),
				description: "Integrated Camera (V4L2)".to_string(),
			})
		);

		let mut screen = WEBCAM;
		screen[1] = ("media.role", "Screen");
		assert_eq!(node(&screen), None);
		let mut sink = WEBCAM;
		sink[0] = ("media.class", "Video/Sink");
		assert_eq!(node(&sink), None);
		// The name is the id's only stable part, so a node without one is unusable.
		assert_eq!(node(&WEBCAM[..2]), None);
	}

	#[test]
	fn a_camera_without_a_description_falls_back_to_its_nick_then_name() {
		let libcamera = [
			("media.class", "Video/Source"),
			("media.role", "Camera"),
			("node.name", "libcamera_input./base/soc/i2c0mux/i2c@1/imx708@1a"),
			("node.nick", "imx708"),
		];
		assert_eq!(node(&libcamera).unwrap().description, "imx708");
		assert_eq!(
			node(&libcamera[..3]).unwrap().description,
			"libcamera_input./base/soc/i2c0mux/i2c@1/imx708@1a"
		);
	}

	#[test]
	fn resolve_finds_the_named_camera_or_fails() {
		let nodes = [node(&WEBCAM).unwrap()];
		assert_eq!(
			resolve(&nodes, Some("v4l2_input.pci-0000_00_14.0-usb-0_4_1.0")).unwrap(),
			Some(62)
		);
		assert_eq!(resolve(&nodes, None).unwrap(), None);
		let error = resolve(&nodes, Some("missing")).unwrap_err().to_string();
		assert!(error.contains("v4l2_input.pci-0000_00_14.0-usb-0_4_1.0"), "{error}");
		assert!(matches!(resolve(&[], None), Err(Error::SourceUnavailable(_))));
	}

	/// Lists the PipeWire cameras over the session socket. Needs no camera and
	/// never turns one on, so a host without PipeWire just lists nothing.
	#[tokio::test]
	async fn lists_cameras_over_the_session_socket() {
		if ashpd::is_sandboxed() {
			eprintln!("skipping: listing inside a sandbox would ask the camera portal");
			return;
		}
		let cameras = cameras().await.expect("listing PipeWire cameras");
		for camera in &cameras {
			assert!(camera.id.starts_with("pipewire:"), "{}", camera.id);
			assert!(!camera.name.is_empty());
		}
		eprintln!("PipeWire cameras: {cameras:?}");
	}

	/// Captures a few frames from each PipeWire camera over the session socket.
	/// Ignored because it turns the cameras on:
	/// `cargo test -p moq-video --features pipewire pipewire_camera -- --ignored`.
	#[tokio::test]
	#[ignore = "turns on every PipeWire camera on the host"]
	async fn pipewire_camera_captures_frames() {
		if ashpd::is_sandboxed() {
			eprintln!("skipping: capturing inside a sandbox would ask the camera portal");
			return;
		}
		let cameras = cameras().await.expect("listing PipeWire cameras");
		if cameras.is_empty() {
			eprintln!("skipping: no PipeWire camera");
			return;
		}

		let mut captured = 0;
		for camera in cameras {
			let config = Config {
				source: camera.source(),
				..Default::default()
			};
			let mut stream = match crate::capture::open(&config).await {
				Ok(stream) => stream,
				// An IR camera offers only GRAY8, which this backend does not convert.
				Err(error) => {
					eprintln!("{}: not captured: {error}", camera.id);
					continue;
				}
			};
			assert_eq!(stream.label(), camera.id);
			assert!(stream.width() >= 2 && stream.width().is_multiple_of(2), "bad width");
			assert!(stream.height() >= 2 && stream.height().is_multiple_of(2), "bad height");
			for i in 0..5 {
				let frame = stream
					.read()
					.await
					.unwrap_or_else(|error| panic!("{}: read frame {i}: {error}", camera.id))
					.unwrap_or_else(|| panic!("{}: no frame {i}", camera.id));
				assert_eq!(frame.surface.width(), stream.width());
				assert_eq!(frame.surface.height(), stream.height());
			}
			eprintln!(
				"{}: captured 5 frames at {}x{}, {:?} fps, color {:?}",
				camera.id,
				stream.width(),
				stream.height(),
				stream.framerate(),
				stream.color()
			);
			captured += 1;
		}
		assert!(captured > 0, "no PipeWire camera could be captured");

		// `pipewire` alone leaves the choice to the session manager.
		let config = Config {
			source: crate::capture::Source::Camera(Some(PIPEWIRE.to_string())),
			..Default::default()
		};
		let mut stream = crate::capture::open(&config).await.expect("default PipeWire camera");
		assert_eq!(stream.label(), PIPEWIRE);
		stream
			.read()
			.await
			.expect("read")
			.expect("a frame from the default camera");
	}
}
