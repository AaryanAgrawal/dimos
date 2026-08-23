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

"""dimos LCM wire messages to Foxglove messages, dispatched by the generated struct class."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
from math import isfinite
import struct
from typing import Any, cast

from dimos_lcm.geometry_msgs import PoseStamped
from dimos_lcm.nav_msgs import OccupancyGrid, Odometry, Path
from dimos_lcm.sensor_msgs import CameraInfo, Image, PointCloud2
from dimos_lcm.tf2_msgs import TFMessage
from foxglove import channels, messages

NS_PER_S = 1_000_000_000

# ROS PointField ids are not foxglove's: https://docs.foxglove.dev/docs/visualization/message-schemas/packed-element-field
_NUMERIC_TYPES = {
    1: messages.PackedElementFieldNumericType.Int8,
    2: messages.PackedElementFieldNumericType.Uint8,
    3: messages.PackedElementFieldNumericType.Int16,
    4: messages.PackedElementFieldNumericType.Uint16,
    5: messages.PackedElementFieldNumericType.Int32,
    6: messages.PackedElementFieldNumericType.Uint32,
    7: messages.PackedElementFieldNumericType.Float32,
    8: messages.PackedElementFieldNumericType.Float64,
}

_IDENTITY_POSE = messages.Pose(
    position=messages.Vector3(x=0.0, y=0.0, z=0.0),
    orientation=messages.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
)

_MAX_JSON_ITEMS = 64  # longer arrays belong in a 3D panel, so JSON carries only their length

# These nav debug-viz types generate no struct of their own; each borrows one, per its own docstring.
_WIRE_ALIASES: dict[str, type[Any]] = {
    "nav_msgs.ContourPolygons3D": PointCloud2,
    "nav_msgs.GraphNodes3D": Path,
    "nav_msgs.LineSegments3D": Path,
}


def timestamp(ns: int) -> messages.Timestamp:
    """Unix nanoseconds as a foxglove Timestamp."""
    return messages.Timestamp(ns // NS_PER_S, ns % NS_PER_S)


def stamp_ns(msg: Any, recv_ns: int) -> int:
    """Header stamp in unix ns; sec == 0 means the publisher had no clock, so use receive time."""
    header = getattr(msg, "header", None)
    if header is None or header.stamp.sec == 0:
        return recv_ns
    return int(header.stamp.sec) * NS_PER_S + int(header.stamp.nsec)


def wire_type(lcm_type: type[Any]) -> type[Any] | None:
    """The generated dimos_lcm struct for a resolved msg class: a decode that keeps raw buffers."""
    msg_name: str | None = getattr(lcm_type, "msg_name", None)
    if msg_name is None:
        return None
    if msg_name in _WIRE_ALIASES:
        return _WIRE_ALIASES[msg_name]
    module_name, _, class_name = msg_name.rpartition(".")
    try:
        module = importlib.import_module(f"dimos_lcm.{module_name}")
    except ImportError:
        return None
    return getattr(module, class_name, None)


def _vector3(vector: Any) -> messages.Vector3:
    return messages.Vector3(x=vector.x, y=vector.y, z=vector.z)


def _quaternion(rotation: Any) -> messages.Quaternion:
    return messages.Quaternion(x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w)


def _pose(pose: Any) -> messages.Pose:
    return messages.Pose(
        position=_vector3(pose.position), orientation=_quaternion(pose.orientation)
    )


def _stamp(stamp: Any, fallback: messages.Timestamp) -> messages.Timestamp:
    if stamp.sec == 0:
        return fallback
    return messages.Timestamp(stamp.sec, stamp.nsec)


def _point_cloud(msg: PointCloud2, ts: messages.Timestamp) -> messages.PointCloud:
    """Pass the packed point buffer through untouched; only the envelope is translated."""
    return messages.PointCloud(
        timestamp=ts,
        frame_id=msg.header.frame_id,
        pose=_IDENTITY_POSE,
        point_stride=msg.point_step,
        fields=[
            messages.PackedElementField(
                name=field.name,
                offset=field.offset,
                type=_NUMERIC_TYPES.get(
                    field.datatype, messages.PackedElementFieldNumericType.Unknown
                ),
            )
            for field in msg.fields
        ],
        data=msg.data,
    )


def _raw_image(msg: Image, ts: messages.Timestamp) -> messages.RawImage:
    """dimos writes ROS encoding names, which foxglove RawImage accepts verbatim."""
    if msg.encoding == "jpeg":
        raise ValueError("got jpeg image data, want raw pixels: drop the jpeg_lcm remap")
    return messages.RawImage(
        timestamp=ts,
        frame_id=msg.header.frame_id,
        width=msg.width,
        height=msg.height,
        encoding=msg.encoding,
        step=msg.step,
        data=msg.data,
    )


def _camera_calibration(msg: CameraInfo, ts: messages.Timestamp) -> messages.CameraCalibration:
    return messages.CameraCalibration(
        timestamp=ts,
        frame_id=msg.header.frame_id,
        width=msg.width,
        height=msg.height,
        distortion_model=msg.distortion_model,
        D=list(msg.D),
        K=list(msg.K),
        R=list(msg.R),
        P=list(msg.P),
    )


def _frame_transforms(msg: TFMessage, ts: messages.Timestamp) -> messages.FrameTransforms:
    """TFMessage has no envelope header, so every transform carries its own stamp."""
    return messages.FrameTransforms(
        transforms=[
            messages.FrameTransform(
                timestamp=_stamp(transform.header.stamp, ts),
                parent_frame_id=transform.header.frame_id,
                child_frame_id=transform.child_frame_id,
                translation=_vector3(transform.transform.translation),
                rotation=_quaternion(transform.transform.rotation),
            )
            for transform in msg.transforms
        ]
    )


def _pose_in_frame(msg: PoseStamped, ts: messages.Timestamp) -> messages.PoseInFrame:
    return messages.PoseInFrame(timestamp=ts, frame_id=msg.header.frame_id, pose=_pose(msg.pose))


def _odometry_pose(msg: Odometry, ts: messages.Timestamp) -> messages.PoseInFrame:
    return messages.PoseInFrame(
        timestamp=ts, frame_id=msg.header.frame_id, pose=_pose(msg.pose.pose)
    )


def _poses_in_frame(msg: Path, ts: messages.Timestamp) -> messages.PosesInFrame:
    return messages.PosesInFrame(
        timestamp=ts,
        frame_id=msg.header.frame_id,
        poses=[_pose(stamped.pose) for stamped in msg.poses],
    )


def _grid(msg: OccupancyGrid, ts: messages.Timestamp) -> messages.Grid:
    """One int8 cost per cell, so the decoded cell array is the Grid buffer."""
    info = msg.info
    return messages.Grid(
        timestamp=ts,
        frame_id=msg.header.frame_id,
        pose=_pose(info.origin),
        column_count=info.width,
        cell_size=messages.Vector2(x=info.resolution, y=info.resolution),
        row_stride=info.width,
        cell_stride=1,
        fields=[
            messages.PackedElementField(
                name="cost", offset=0, type=messages.PackedElementFieldNumericType.Int8
            )
        ],
        data=struct.pack(f"{len(msg.data)}b", *msg.data),
    )


def _jsonable(value: Any) -> Any:
    """Reduce one wire field to JSON; buffers and long arrays become their length."""
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return len(value) if len(value) > _MAX_JSON_ITEMS else [_jsonable(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None  # json.dumps writes bare NaN/Infinity, which a strict JSON parser rejects
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {name: _jsonable(getattr(value, name)) for name in type(value).__slots__}


def to_json(msg: Any) -> dict[str, Any]:
    """Every wire field as JSON, so Plot and Raw Messages panels work on any decodable topic."""
    return cast("dict[str, Any]", _jsonable(msg))


@dataclass(frozen=True)
class Mapping:
    """The foxglove channel class advertised for a wire type and the function that fills it."""

    channel: type[Any]
    convert: Callable[[Any, messages.Timestamp], Any]


MAPPINGS: dict[type[Any], Mapping] = {
    CameraInfo: Mapping(channels.CameraCalibrationChannel, _camera_calibration),
    Image: Mapping(channels.RawImageChannel, _raw_image),
    OccupancyGrid: Mapping(channels.GridChannel, _grid),
    Odometry: Mapping(channels.PoseInFrameChannel, _odometry_pose),
    Path: Mapping(channels.PosesInFrameChannel, _poses_in_frame),
    PointCloud2: Mapping(channels.PointCloudChannel, _point_cloud),
    PoseStamped: Mapping(channels.PoseInFrameChannel, _pose_in_frame),
    TFMessage: Mapping(channels.FrameTransformsChannel, _frame_transforms),
}
