# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Foxglove bridge converters and its live WebSocket path."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import struct
import threading
import time
from types import SimpleNamespace
from typing import Any

from dimos_lcm.geometry_msgs import (
    Point,
    Pose,
    PoseStamped,
    PoseWithCovariance,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
    TwistWithCovariance,
    Vector3,
)
from dimos_lcm.nav_msgs import MapMetaData, OccupancyGrid, Odometry, Path
from dimos_lcm.sensor_msgs import CameraInfo, Image, PointCloud2, PointField, RegionOfInterest
from dimos_lcm.std_msgs import Header, Time
from dimos_lcm.tf2_msgs import TFMessage
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
import numpy as np
import pytest
from websockets.sync.client import connect

from dimos.msgs.helpers import resolve_msg_type
from dimos.protocol.pubsub.impl.lcmpubsub import LCMPubSubBase
from dimos.utils.testing.waiting import wait_until
from dimos.visualization.foxglove import bridge, converters
from dimos.visualization.foxglove.bridge import FoxgloveBridgeModule, _Slot, _TeleopListener

_TEST_BUS = "udpm://239.255.76.68:7668?ttl=0"  # never the default bus a live stack runs on
_E2E_PORT = 18765
_TELEOP_PORT = 18766
_SUBPROTOCOL = "foxglove.sdk.v1"  # https://github.com/foxglove/ws-protocol/blob/main/docs/spec.md
_LIDAR_CHANNEL = "/lidar#sensor_msgs.PointCloud2"
_MESSAGE_DATA_OP = 1
_SUBSCRIPTION_ID = 7
_TWIST_CHANNEL = 11
_POINT_CHANNEL = 12
_STRING_CHANNEL = 13
_SHARED_CHANNEL = 1  # every client numbers its own channels, so two of them pick this one
_STAMP_NS = 12 * converters.NS_PER_S + 34
_TIMEOUT_S = 10.0
_POLL_S = 0.2

_CLOUD_BUFFER = np.arange(8, dtype=np.float32).tobytes()
_TIMESTAMP = converters.timestamp(_STAMP_NS)
_DRIVE = {"linear": {"x": 0.4, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": -0.75}}
_CLICK = {
    "header": {"seq": 0, "stamp": {"sec": 5, "nsec": 250_000_000}, "frame_id": "map"},
    "point": {"x": 1.5, "y": -2.5, "z": 0.0},
}


def _header(sec: int = 12, nsec: int = 34, frame_id: str = "world") -> Header:
    header = Header()
    header.seq = 0
    header.frame_id = frame_id
    header.stamp = Time()
    header.stamp.sec = sec
    header.stamp.nsec = nsec
    return header


def _pose(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> Pose:
    pose = Pose()
    pose.position = Point()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation = Quaternion()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 0, 0, 0, 1.0
    return pose


def _vector3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    vector = Vector3()
    vector.x, vector.y, vector.z = x, y, z
    return vector


def _wire(msg: Any) -> Any:
    """Round-trip a constructed struct through the LCM wire, so tests see decoded values."""
    return type(msg).lcm_decode(msg.lcm_encode())


def _parsed(msg: Any) -> Any:
    """Parse a foxglove message back through its own protobuf schema so fields can be asserted."""
    schema = type(msg).get_schema()
    pool = descriptor_pool.DescriptorPool()
    for proto in descriptor_pb2.FileDescriptorSet.FromString(schema.data).file:
        pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name)).FromString(
        msg.encode()
    )


def _point_cloud2() -> PointCloud2:
    msg = PointCloud2()
    msg.header = _header(frame_id="lidar")
    msg.height, msg.width = 1, 2
    msg.point_step, msg.row_step = 16, 32
    msg.is_bigendian, msg.is_dense = False, True
    msg.fields = []
    for name, offset in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12)):
        field = PointField()
        field.name, field.offset, field.datatype, field.count = name, offset, 7, 1
        msg.fields.append(field)
    msg.fields_length = 4
    msg.data, msg.data_length = _CLOUD_BUFFER, len(_CLOUD_BUFFER)
    return msg


def _image(encoding: str = "bgr8", step: int = 9) -> Image:
    msg = Image()
    msg.header = _header(frame_id="camera_optical")
    msg.height, msg.width = 2, 3
    msg.encoding, msg.is_bigendian, msg.step = encoding, False, step
    msg.data, msg.data_length = bytes(range(18)), 18
    return msg


def _camera_info() -> CameraInfo:
    msg = CameraInfo()
    msg.header = _header(frame_id="camera_optical")
    msg.height, msg.width = 2, 3
    msg.distortion_model = "plumb_bob"
    msg.D, msg.D_length = [0.0] * 5, 5
    msg.K = [1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.P = [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.binning_x = msg.binning_y = 0
    msg.roi = RegionOfInterest()
    msg.roi.x_offset = msg.roi.y_offset = msg.roi.height = msg.roi.width = 0
    msg.roi.do_rectify = False
    return msg


def _tf_message() -> TFMessage:
    stamped = TransformStamped()
    stamped.header = _header(sec=7, nsec=8, frame_id="world")
    stamped.child_frame_id = "base_link"
    stamped.transform = Transform()
    stamped.transform.translation = _vector3(1.0, 2.0, 3.0)
    stamped.transform.rotation = _pose().orientation
    msg = TFMessage()
    msg.transforms, msg.transforms_length = [stamped], 1
    return msg


def _pose_stamped(x: float = 1.0) -> PoseStamped:
    msg = PoseStamped()
    msg.header = _header(frame_id="odom")
    msg.pose = _pose(x)
    return msg


def _odometry() -> Odometry:
    msg = Odometry()
    msg.header = _header(frame_id="odom")
    msg.child_frame_id = "base_link"
    msg.pose = PoseWithCovariance()
    msg.pose.pose, msg.pose.covariance = _pose(), [0.0] * 36
    msg.twist = TwistWithCovariance()
    msg.twist.twist = Twist()
    msg.twist.twist.linear, msg.twist.twist.angular = _vector3(), _vector3()
    msg.twist.covariance = [0.0] * 36
    return msg


def _path() -> Path:
    msg = Path()
    msg.header = _header(frame_id="map")
    msg.poses = [_pose_stamped(1.0), _pose_stamped(2.0)]
    msg.poses_length = 2
    return msg


def _occupancy_grid() -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header = _header(frame_id="map")
    msg.info = MapMetaData()
    msg.info.map_load_time = Time()
    msg.info.map_load_time.sec = msg.info.map_load_time.nsec = 0
    msg.info.resolution, msg.info.width, msg.info.height = 0.5, 3, 2
    msg.info.origin = _pose(0.0, 0.0, 0.0)
    msg.data, msg.data_length = [-1, 0, 100, 50, 0, -1], 6
    return msg


def test_point_cloud_forwards_the_packed_buffer() -> None:
    """A PointCloud2's point buffer, stride and field layout reach Foxglove unchanged."""
    cloud = _parsed(converters._point_cloud(_wire(_point_cloud2()), _TIMESTAMP))

    assert cloud.frame_id == "lidar"
    assert cloud.point_stride == 16
    assert cloud.data == _CLOUD_BUFFER
    assert [(f.name, f.offset, f.type) for f in cloud.fields] == [
        ("x", 0, 7),
        ("y", 4, 7),
        ("z", 8, 7),
        ("intensity", 12, 7),
    ]
    assert (cloud.timestamp.seconds, cloud.timestamp.nanos) == (12, 34)


def test_raw_image_forwards_pixels_and_step() -> None:
    """An Image's pixel buffer, row step and ROS encoding name reach Foxglove unchanged."""
    image = _parsed(converters._raw_image(_wire(_image()), _TIMESTAMP))

    assert (image.width, image.height, image.step) == (3, 2, 9)
    assert image.encoding == "bgr8"
    assert image.data == bytes(range(18))
    assert image.frame_id == "camera_optical"


def test_jpeg_image_is_refused() -> None:
    """A jpeg-compressed Image is refused rather than served as raw pixels."""
    with pytest.raises(ValueError, match="want raw pixels"):
        converters._raw_image(_wire(_image(encoding="jpeg", step=0)), _TIMESTAMP)


def test_a_big_endian_buffer_is_refused() -> None:
    """Big-endian points and pixels are refused rather than served as little-endian bytes."""
    cloud = _point_cloud2()
    cloud.is_bigendian = True
    with pytest.raises(ValueError, match="want little-endian"):
        converters._point_cloud(_wire(cloud), _TIMESTAMP)

    image = _image()
    image.is_bigendian = 1
    with pytest.raises(ValueError, match="want little-endian"):
        converters._raw_image(_wire(image), _TIMESTAMP)


def test_camera_calibration_carries_the_intrinsics() -> None:
    """CameraInfo intrinsics, rectification and projection reach Foxglove as CameraCalibration."""
    calibration = _parsed(converters._camera_calibration(_wire(_camera_info()), _TIMESTAMP))

    assert (calibration.width, calibration.height) == (3, 2)
    assert calibration.distortion_model == "plumb_bob"
    assert list(calibration.K) == [1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0]
    assert list(calibration.P) == [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert list(calibration.D) == [0.0] * 5


def test_frame_transforms_keep_each_transform_stamp() -> None:
    """Every transform in a TFMessage becomes a FrameTransform carrying its own stamp."""
    transforms = _parsed(converters._frame_transforms(_wire(_tf_message()), _TIMESTAMP)).transforms

    assert len(transforms) == 1
    assert (transforms[0].parent_frame_id, transforms[0].child_frame_id) == ("world", "base_link")
    assert (transforms[0].translation.x, transforms[0].translation.z) == (1.0, 3.0)
    assert transforms[0].rotation.w == 1.0
    assert (transforms[0].timestamp.seconds, transforms[0].timestamp.nanos) == (7, 8)


def test_pose_stamped_becomes_a_pose_in_frame() -> None:
    """A PoseStamped's frame and pose reach Foxglove as PoseInFrame."""
    pose_in_frame = _parsed(converters._pose_in_frame(_wire(_pose_stamped()), _TIMESTAMP))

    assert pose_in_frame.frame_id == "odom"
    assert (pose_in_frame.pose.position.x, pose_in_frame.pose.position.z) == (1.0, 3.0)
    assert pose_in_frame.pose.orientation.w == 1.0


def test_odometry_publishes_only_its_pose() -> None:
    """An Odometry contributes its pose; twist and covariance have no Foxglove home."""
    pose_in_frame = _parsed(converters._odometry_pose(_wire(_odometry()), _TIMESTAMP))

    assert pose_in_frame.frame_id == "odom"
    assert (pose_in_frame.pose.position.x, pose_in_frame.pose.position.y) == (1.0, 2.0)


def test_path_becomes_poses_in_frame() -> None:
    """A Path's poses reach Foxglove in order under the path's own frame."""
    poses = _parsed(converters._poses_in_frame(_wire(_path()), _TIMESTAMP))

    assert poses.frame_id == "map"
    assert [pose.position.x for pose in poses.poses] == [1.0, 2.0]


def test_grid_packs_one_int8_cost_per_cell() -> None:
    """OccupancyGrid costs reach Foxglove as a one-byte-per-cell Grid buffer."""
    grid = _parsed(converters._grid(_wire(_occupancy_grid()), _TIMESTAMP))

    assert (grid.column_count, grid.row_stride, grid.cell_stride) == (3, 3, 1)
    assert (grid.cell_size.x, grid.cell_size.y) == (0.5, 0.5)
    assert grid.data == np.array([-1, 0, 100, 50, 0, -1], dtype=np.int8).tobytes()
    assert [(f.name, f.offset, f.type) for f in grid.fields] == [("cost", 0, 2)]


def test_stamp_falls_back_to_receive_time_without_a_publisher_clock() -> None:
    """A header stamp of zero seconds means no publisher clock, so receive time is logged."""
    assert converters.stamp_ns(_wire(_point_cloud2()), 99) == _STAMP_NS

    msg = _point_cloud2()
    msg.header = _header(sec=0, nsec=0)
    assert converters.stamp_ns(_wire(msg), 99) == 99


def test_json_summarizes_buffers_and_long_arrays() -> None:
    """Untyped topics serve every wire field as JSON, with buffers reduced to their length."""
    fields = converters.to_json(_wire(_point_cloud2()))

    assert fields["data"] == 32
    assert fields["point_step"] == 16
    assert fields["header"] == {"seq": 0, "stamp": {"sec": 12, "nsec": 34}, "frame_id": "lidar"}
    assert fields["fields"][0] == {"name": "x", "offset": 0, "datatype": 7, "count": 1}
    assert converters._jsonable(list(range(converters._MAX_JSON_ITEMS + 1))) == 65
    assert converters._jsonable(float("nan")) is None


def test_wire_type_resolves_the_generated_struct() -> None:
    """A resolved dimos msg class maps to its own generated struct, or to the one it borrows."""
    assert converters.wire_type(resolve_msg_type("sensor_msgs.PointCloud2")) is PointCloud2
    assert converters.wire_type(resolve_msg_type("nav_msgs.GraphNodes3D")) is Path
    assert converters.wire_type(resolve_msg_type("sensor_msgs.RobotState")) is None


def test_the_bridge_binds_loopback_unless_asked() -> None:
    """The default bind is loopback, so teleop reaches the robot from this host only."""
    module = FoxgloveBridgeModule(pubsubs=[LCMPubSubBase(url=_TEST_BUS)])
    asked = FoxgloveBridgeModule(pubsubs=[LCMPubSubBase(url=_TEST_BUS)], host="0.0.0.0")
    try:
        assert module.host == "127.0.0.1"
        assert asked.host == "0.0.0.0"
    finally:
        module.stop()
        asked.stop()


def test_latest_wins_replaces_the_undrained_payload() -> None:
    """A payload landing before a worker drains the slot replaces it and counts one drop."""
    module = FoxgloveBridgeModule(pubsubs=[LCMPubSubBase(url=_TEST_BUS)])
    try:
        slot = _Slot(channel=None, decode=bytes, convert=None)
        with module._cond:
            module._offer(slot, b"stale")
            module._offer(slot, b"fresh")
        assert (slot.payload, slot.dropped, len(module._pending)) == (b"fresh", 1, 1)
    finally:
        module.stop()


@contextmanager
def _bridge(port: int) -> Iterator[FoxgloveBridgeModule]:
    """A started bridge bound to a private port and the isolated test bus."""
    module = FoxgloveBridgeModule(
        pubsubs=[LCMPubSubBase(url=_TEST_BUS)], host="127.0.0.1", port=port
    )
    module.start()
    try:
        yield module
    finally:
        module.stop()


def _publish_until(ws: Any, publish: Callable[[], None], match: Callable[[Any], bool]) -> Any:
    """Republish until a matching server frame arrives; multicast may drop the first datagram."""
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        publish()
        try:
            frame = ws.recv(timeout=_POLL_S)
        except TimeoutError:
            continue
        parsed = json.loads(frame) if isinstance(frame, str) else frame
        if match(parsed):
            return parsed
    raise TimeoutError(f"foxglove server sent no matching frame within {_TIMEOUT_S}s")


def _advertised_channel(ws: Any, publish: Callable[[], None]) -> Any:
    """Republish until the server advertises, and return the advertised channel."""

    def advertised(frame: Any) -> bool:
        return isinstance(frame, dict) and frame["op"] == "advertise"

    return _publish_until(ws, publish, advertised)["channels"][0]


def _subscribed_frame(ws: Any, channel_id: int, publish: Callable[[], None]) -> Any:
    """Subscribe to a channel and return the first binary MessageData frame it delivers."""
    subscription = {"id": _SUBSCRIPTION_ID, "channelId": channel_id}
    ws.send(json.dumps({"op": "subscribe", "subscriptions": [subscription]}))
    return _publish_until(ws, publish, lambda frame: isinstance(frame, bytes))


def _count_conversions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Swap in a PointCloud mapping that records every conversion it performs."""
    conversions: list[int] = []
    mapping = converters.MAPPINGS[PointCloud2]

    def counted(msg: Any, ts: Any) -> Any:
        conversions.append(1)
        return mapping.convert(msg, ts)

    monkeypatch.setitem(
        converters.MAPPINGS, PointCloud2, converters.Mapping(mapping.channel, counted)
    )
    return conversions


def test_bridge_advertises_a_topic_before_it_decodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A topic advertises from its type suffix alone and decodes only once a client subscribes."""
    conversions = _count_conversions(monkeypatch)
    payload = _point_cloud2().lcm_encode()
    publisher = LCMPubSubBase(url=_TEST_BUS)
    try:
        with (
            _bridge(_E2E_PORT),
            connect(f"ws://127.0.0.1:{_E2E_PORT}", subprotocols=[_SUBPROTOCOL]) as ws,
        ):

            def publish() -> None:
                publisher.publish(_LIDAR_CHANNEL, payload)

            channel = _advertised_channel(ws, publish)
            assert (channel["topic"], channel["schemaName"]) == ("/lidar", "foxglove.PointCloud")
            assert conversions == []
            frame = _subscribed_frame(ws, channel["id"], publish)
    finally:
        publisher.stop()

    assert frame[0] == _MESSAGE_DATA_OP
    assert struct.unpack("<IQ", frame[1:13]) == (_SUBSCRIPTION_ID, _STAMP_NS)
    assert conversions
    assert not [t for t in threading.enumerate() if t.name.startswith("foxglove-bridge")]


@contextmanager
def _capture(stream: Any) -> Iterator[list[Any]]:
    """Collect what the bridge publishes on one Out for as long as the test needs it."""
    received: list[Any] = []
    unsubscribe = stream.subscribe(received.append)
    try:
        yield received
    finally:
        unsubscribe()


def _client(port: int) -> Any:
    """A websocket client speaking the foxglove protocol, as a Foxglove panel would."""
    return connect(f"ws://127.0.0.1:{port}", subprotocols=[_SUBPROTOCOL])


def _advertise(ws: Any, channel_id: int, schema: str) -> None:
    """Advertise one client channel on a topic of the client's own choosing."""
    channel = {"id": channel_id, "topic": "/anything", "encoding": "json", "schemaName": schema}
    ws.send(json.dumps({"op": "advertise", "channels": [channel]}))


def _client_publish(ws: Any, channel_id: int, fields: dict[str, Any]) -> None:
    """Publish one JSON message from the client on a channel it advertised."""
    ws.send(struct.pack("<BI", _MESSAGE_DATA_OP, channel_id) + json.dumps(fields).encode())


def _wait_for(received: list[Any], count: int) -> None:
    """Block until the bridge has published `count` messages, so no test sleeps on a fixed delay."""
    wait_until(
        lambda: len(received) >= count,
        timeout=_TIMEOUT_S,
        message=f"want {count} messages on the Out",
    )


def test_teleop_twist_reaches_tele_cmd_vel() -> None:
    """A Teleop panel Twist reaches tele_cmd_vel with every component it was published with."""
    with (
        _bridge(_TELEOP_PORT) as module,
        _capture(module.tele_cmd_vel) as received,
        _client(_TELEOP_PORT) as ws,
    ):
        _advertise(ws, _TWIST_CHANNEL, "geometry_msgs/Twist")
        _client_publish(ws, _TWIST_CHANNEL, _DRIVE)
        _wait_for(received, 1)

        twist = received[0]
        assert (twist.linear.x, twist.linear.y, twist.linear.z) == (0.4, 0.0, 0.0)
        assert (twist.angular.x, twist.angular.y, twist.angular.z) == (0.0, 0.0, -0.75)


def test_publish_tool_click_reaches_clicked_point() -> None:
    """A 3D panel Publish click reaches clicked_point in the frame and at the stamp it carried."""
    with (
        _bridge(_TELEOP_PORT) as module,
        _capture(module.clicked_point) as received,
        _client(_TELEOP_PORT) as ws,
    ):
        _advertise(ws, _POINT_CHANNEL, "geometry_msgs/PointStamped")
        _client_publish(ws, _POINT_CHANNEL, _CLICK)
        _wait_for(received, 1)

        point = received[0]
        assert (point.x, point.y, point.z) == (1.5, -2.5, 0.0)
        assert (point.frame_id, point.ts) == ("map", 5.25)


def test_unadvertise_stops_the_robot() -> None:
    """A teleop client that unadvertises mid-drive leaves a zero Twist, not its last command."""
    with (
        _bridge(_TELEOP_PORT) as module,
        _capture(module.tele_cmd_vel) as received,
        _client(_TELEOP_PORT) as ws,
    ):
        _advertise(ws, _TWIST_CHANNEL, "geometry_msgs/Twist")
        _client_publish(ws, _TWIST_CHANNEL, _DRIVE)
        _wait_for(received, 1)
        ws.send(json.dumps({"op": "unadvertise", "channelIds": [_TWIST_CHANNEL]}))
        _wait_for(received, 2)

        assert not received[0].is_zero()
        assert received[1].is_zero()


def test_disconnect_stops_the_robot() -> None:
    """A teleop client that drops its connection mid-drive leaves a zero Twist behind."""
    with _bridge(_TELEOP_PORT) as module, _capture(module.tele_cmd_vel) as received:
        with _client(_TELEOP_PORT) as ws:
            _advertise(ws, _TWIST_CHANNEL, "geometry_msgs/Twist")
            _client_publish(ws, _TWIST_CHANNEL, _DRIVE)
            _wait_for(received, 1)
        _wait_for(received, 2)

        assert not received[0].is_zero()
        assert received[1].is_zero()


def test_unroutable_schema_publishes_nothing() -> None:
    """A schema no Out serves is refused, so only the routable channel reaches the bus."""
    with (
        _bridge(_TELEOP_PORT) as module,
        _capture(module.tele_cmd_vel) as twists,
        _capture(module.clicked_point) as points,
        _client(_TELEOP_PORT) as ws,
    ):
        _advertise(ws, _STRING_CHANNEL, "std_msgs/String")
        _advertise(ws, _TWIST_CHANNEL, "geometry_msgs/Twist")
        _client_publish(ws, _STRING_CHANNEL, {"data": "drive"})
        _client_publish(ws, _TWIST_CHANNEL, _DRIVE)
        _wait_for(twists, 1)

        assert (len(twists), len(points)) == (1, 0)
        assert twists[0].linear.x == 0.4


@contextmanager
def _listener() -> Iterator[tuple[_TeleopListener, FoxgloveBridgeModule]]:
    """A listener on a serverless bridge, so its callbacks run on the test's own thread."""
    module = FoxgloveBridgeModule(pubsubs=[LCMPubSubBase(url=_TEST_BUS)])
    try:
        yield _TeleopListener(module), module
    finally:
        module.stop()


def _sdk_client(client_id: int) -> Any:
    """The id-only Client the SDK hands to every listener callback."""
    return SimpleNamespace(id=client_id)


def _advertise_to(listener: _TeleopListener, client_id: int, channel: int, schema: str) -> None:
    """Advertise one client channel, as the SDK does when a panel starts publishing."""
    listener.on_client_advertise(
        _sdk_client(client_id), SimpleNamespace(id=channel, schema_name=schema, encoding="json")
    )


def _publish_to(
    listener: _TeleopListener, client_id: int, channel: int, fields: dict[str, Any]
) -> None:
    """Hand the listener one client message, as the SDK does when a panel publishes."""
    listener.on_message_data(_sdk_client(client_id), channel, json.dumps(fields).encode())


def test_two_clients_may_pick_the_same_channel_id() -> None:
    """A channel id is unique per client, so one client's channel is never routed as another's."""
    with (
        _listener() as (listener, module),
        _capture(module.tele_cmd_vel) as twists,
        _capture(module.clicked_point) as points,
    ):
        _advertise_to(listener, 1, _SHARED_CHANNEL, "geometry_msgs/Twist")
        _advertise_to(listener, 2, _SHARED_CHANNEL, "geometry_msgs/PointStamped")
        _publish_to(listener, 1, _SHARED_CHANNEL, _DRIVE)
        _publish_to(listener, 2, _SHARED_CHANNEL, _CLICK)
        listener.on_client_unadvertise(_sdk_client(2), _SHARED_CHANNEL)
        _publish_to(listener, 1, _SHARED_CHANNEL, _DRIVE)

        assert [twist.linear.x for twist in twists] == [0.4, 0.4]
        assert [(point.x, point.y) for point in points] == [(1.5, -2.5)]


def test_only_a_driving_channel_stops_the_robot() -> None:
    """A teleop panel that closes without driving cancels no goal; one that was driving stops."""
    with _listener() as (listener, module), _capture(module.tele_cmd_vel) as twists:
        _advertise_to(listener, 1, _TWIST_CHANNEL, "geometry_msgs/Twist")
        listener.on_client_unadvertise(_sdk_client(1), _TWIST_CHANNEL)
        assert twists == []

        _advertise_to(listener, 1, _TWIST_CHANNEL, "geometry_msgs/Twist")
        _publish_to(listener, 1, _TWIST_CHANNEL, _DRIVE)
        listener.on_client_unadvertise(_sdk_client(1), _TWIST_CHANNEL)

        assert [twist.is_zero() for twist in twists] == [False, True]


def test_an_unroutable_schema_is_refused() -> None:
    """A client channel the bridge serves no Out for is refused, so its messages never publish."""
    with _listener() as (listener, module), _capture(module.tele_cmd_vel) as twists:
        _advertise_to(listener, 1, _STRING_CHANNEL, "std_msgs/String")
        _publish_to(listener, 1, _STRING_CHANNEL, {"data": "drive"})

        assert twists == []


def test_a_dropped_message_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A panel republishing a payload no converter accepts warns once, not once per message."""
    warnings: list[str] = []
    monkeypatch.setattr(bridge.logger, "warning", lambda event, **kwargs: warnings.append(event))
    with _listener() as (listener, module), _capture(module.tele_cmd_vel) as twists:
        _advertise_to(listener, 1, _TWIST_CHANNEL, "geometry_msgs/Twist")
        _publish_to(listener, 1, _TWIST_CHANNEL, {"linear": {}, "angular": {}})
        _publish_to(listener, 1, _TWIST_CHANNEL, {"linear": {}, "angular": {}})

        assert (warnings, twists) == (["foxglove bridge dropped a client message"], [])


def test_a_click_payload_without_coordinates_is_refused() -> None:
    """A payload with no coordinates is refused; a bare Point still becomes a click."""
    with pytest.raises(ValueError, match="want an object with x, y and z"):
        converters.point_stamped(_DRIVE)
    with pytest.raises(ValueError, match="want an object with x, y and z"):
        converters.point_stamped({"point": {}})

    assert converters.point_stamped({"x": 1.0, "y": 2.0, "z": 3.0}).y == 2.0


def test_a_non_finite_velocity_is_refused() -> None:
    """NaN and infinite velocities are refused rather than scaled onto cmd_vel."""
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="want finite x, y and z"):
            converters.twist({"linear": {"x": value}, "angular": {"z": 0.0}})
