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

"""Foxglove WebSocket bridge serving every pubsub topic to the Foxglove app."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import json
import signal
import sys
import threading
import time
from typing import Any

import foxglove
from foxglove import Channel
from foxglove.websocket import Capability, Client, ClientChannel, ServerListener, WebSocketServer
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.global_config import global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.protocol.pubsub.impl.lcmpubsub import LCMPubSubBase
from dimos.protocol.pubsub.impl.zenohpubsub import ZenohPubSubBase
from dimos.protocol.pubsub.spec import SubscribeAllCapable
from dimos.protocol.service.lcmservice import autoconf
from dimos.utils.logging_config import setup_logger
from dimos.visualization.foxglove import converters

logger = setup_logger()

FOXGLOVE_PORT = 8765
FOXGLOVE_HOST = "0.0.0.0"  # the Foxglove app usually runs on another machine

_WORKERS = 2  # decode/convert pool; the bus thread only fills mailboxes
_TELE_CMD_VEL = "tele_cmd_vel"
_CLICKED_POINT = "clicked_point"


@dataclass
class _Slot:
    """Latest-wins mailbox for one topic: the bus thread writes, a worker publishes."""

    channel: Any
    decode: Callable[[bytes], Any]
    convert: Callable[[Any, Any], Any] | None
    payload: bytes = b""  # empty means nothing pending: an LCM payload always carries a fingerprint
    recv_ns: int = 0
    dropped: int = 0
    owned: bool = False
    warned: bool = False


def _topic_name(topic: Any) -> str:
    """Foxglove topic name: the dimos topic without its type suffix, always rooted at '/'."""
    name: str = topic.pattern
    return name if name.startswith("/") else "/" + name.removeprefix("dimos/")


def _default_pubsubs(config: Any) -> list[SubscribeAllCapable[Any, Any]]:
    """Encoder-free pubsub for the active transport, so callbacks carry undecoded bytes."""
    transport = getattr(config, "transport", None) or global_config.transport
    if transport == "zenoh":
        return [ZenohPubSubBase()]
    return [LCMPubSubBase()]


def _resolve_pubsubs(config: Config) -> list[SubscribeAllCapable[Any, Any]]:
    """Explicit pubsubs win, otherwise follow the active transport."""
    return config.pubsubs or _default_pubsubs(config.g)


@dataclass
class _Route:
    """The bridge Out a client-advertised schema publishes on, and the decoder that fills it."""

    stream: str
    decode: Callable[[dict[str, Any]], Any]
    moving: bool = False
    warned: bool = False


_ROUTES = {
    "Twist": _Route(_TELE_CMD_VEL, converters.twist),
    "PointStamped": _Route(_CLICKED_POINT, converters.point_stamped),
    "Point": _Route(_CLICKED_POINT, converters.point_stamped),
}


def _route(schema_name: str) -> _Route | None:
    """The route for a client's advertised schema; None when the bridge serves no Out for it."""
    for suffix, route in _ROUTES.items():
        if schema_name.endswith(suffix):
            return replace(route)  # a copy per channel: one client's state is not another's
    return None


class _TeleopListener(ServerListener):
    """Republish what the Foxglove Teleop and 3D Publish panels send onto the dimos bus."""

    def __init__(self, bridge: FoxgloveBridgeModule) -> None:
        self._bridge = bridge
        self._routes: dict[tuple[int, int], _Route] = {}  # a channel id is unique per client only

    def on_client_advertise(self, client: Client, channel: ClientChannel) -> None:
        """Accept a client channel whose schema routes to an Out, and refuse the rest."""
        route = _route(channel.schema_name) if channel.encoding == "json" else None
        if route is None:
            logger.warning(
                "foxglove bridge refused client channel",
                schema=channel.schema_name,
                encoding=channel.encoding,
            )
            return
        self._routes[(client.id, channel.id)] = route
        logger.info(
            "foxglove bridge accepted client channel",
            schema=channel.schema_name,
            stream=route.stream,
        )

    def on_message_data(self, client: Client, client_channel_id: int, data: bytes) -> None:
        """Publish one client message on the Out its channel routes to."""
        route = self._routes.get((client.id, client_channel_id))
        if route is None:
            return
        try:
            msg = route.decode(json.loads(data))
            getattr(self._bridge, route.stream).publish(msg)
            route.moving = route.stream == _TELE_CMD_VEL and not msg.is_zero()
        except Exception:
            if route.warned:  # a panel republishes at its own rate, so warn on the first drop only
                return
            route.warned = True
            logger.warning(
                "foxglove bridge dropped a client message", stream=route.stream, exc_info=True
            )

    def on_client_unadvertise(self, client: Client, client_channel_id: int) -> None:
        """Stop the robot when a channel that was driving goes away, so no twist stands."""
        route = self._routes.pop((client.id, client_channel_id), None)
        if route is None or not route.moving:  # a channel that never drove must not cancel a goal
            return
        try:
            self._bridge.tele_cmd_vel.publish(Twist.zero())  # the SDK calls this on disconnect too
        except Exception:
            logger.warning("foxglove bridge could not stop the robot", exc_info=True)


class Config(ModuleConfig):
    """Configuration for FoxgloveBridgeModule."""

    pubsubs: list[SubscribeAllCapable[Any, Any]] = field(default_factory=list)
    host: str = FOXGLOVE_HOST
    port: int = FOXGLOVE_PORT
    max_hz: dict[str, float] = field(default_factory=dict)


class FoxgloveBridgeModule(Module):
    """Serve every topic on the dimos bus to the Foxglove app over its WebSocket protocol."""

    config: Config
    dedicated_worker = True

    clicked_point: Out[PointStamped]
    tele_cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cond = threading.Condition()
        self._pending: deque[_Slot] = deque()
        self._slots: dict[str, _Slot | None] = {}
        self._workers: list[threading.Thread] = []
        self._server: WebSocketServer | None = None
        self._stopping = False
        self._min_intervals: dict[str, float] = {}
        self._last_log: dict[str, float] = {}

    @rpc
    def start(self) -> None:
        super().start()
        self._stopping = False
        self._min_intervals = {t: 1.0 / hz for t, hz in self.config.max_hz.items() if hz > 0}
        self._server = foxglove.start_server(
            name="dimos",
            host=self.config.host,
            port=self.config.port,
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],  # what the panels publish against a non-ROS server
            server_listener=_TeleopListener(self),
        )
        self._start_workers()
        self._subscribe(_resolve_pubsubs(self.config))
        logger.info("foxglove bridge listening", host=self.config.host, port=self.config.port)

    @rpc
    def stop(self) -> None:
        self._stop_workers()
        if self._server is not None:
            self._server.stop()
            self._server = None
        self._close_slots()
        super().stop()

    def _start_workers(self) -> None:
        """Decode off the bus thread so one heavy topic never stalls every other topic."""
        for index in range(_WORKERS):
            worker = threading.Thread(
                target=self._drain, name=f"foxglove-bridge-{index}", daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def _subscribe(self, pubsubs: list[SubscribeAllCapable[Any, Any]]) -> None:
        """Attach to every pubsub and register its teardown."""
        for pubsub in pubsubs:
            if hasattr(pubsub, "start"):
                pubsub.start()
            self.register_disposable(Disposable(pubsub.subscribe_all(self._on_raw)))
            if hasattr(pubsub, "stop"):
                self.register_disposable(Disposable(pubsub.stop))

    def _on_raw(self, data: bytes, topic: Any) -> None:
        """Bus thread: gate on subscribers, then hand the bytes to a worker. Never decode here."""
        name = _topic_name(topic)
        try:
            with self._cond:
                if self._stopping:
                    return
                if name not in self._slots:
                    self._slots[name] = self._build_slot(name, topic)
                slot = self._slots[name]
                # no sink: the channel was advertised from the type suffix, so nothing decodes yet
                if slot is None or not slot.channel.has_sinks() or self._throttled(name):
                    return
                self._offer(slot, data)
        except Exception:
            logger.warning("foxglove bridge dropped a message", topic=name, exc_info=True)

    def _build_slot(self, name: str, topic: Any) -> _Slot | None:
        """Advertise a channel for a topic's wire type; None when the bridge cannot serve it."""
        if topic.lcm_type is None:
            return None
        wire = converters.wire_type(topic.lcm_type)
        if wire is None:
            msg_type = getattr(topic.lcm_type, "msg_name", "")
            logger.warning("foxglove bridge has no LCM struct", topic=name, msg_type=msg_type)
            return None
        mapping = converters.MAPPINGS.get(wire)
        if mapping is None:
            return _Slot(Channel(name, message_encoding="json"), wire.lcm_decode, None)
        return _Slot(mapping.channel(name), wire.lcm_decode, mapping.convert)

    def _throttled(self, name: str) -> bool:
        """Drop a message when its topic is already at its configured max_hz."""
        interval = self._min_intervals.get(name)
        if interval is None:
            return False
        now = time.monotonic()
        if now - self._last_log.get(name, 0.0) < interval:
            return True
        self._last_log[name] = now
        return False

    def _offer(self, slot: _Slot, data: bytes) -> None:
        """Latest-wins: overwrite any unpublished payload and queue the slot at most once."""
        if slot.payload:
            slot.dropped += 1
        slot.payload = data
        slot.recv_ns = time.time_ns()
        if not slot.owned:
            slot.owned = True
            self._pending.append(slot)
            self._cond.notify()

    def _drain(self) -> None:
        """Worker loop: take the newest payload of a queued topic and publish it."""
        slot: _Slot | None = None
        while True:
            with self._cond:
                if slot is not None:
                    self._release(slot)
                while not self._pending and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                slot = self._pending.popleft()
                payload, recv_ns = slot.payload, slot.recv_ns
                slot.payload = b""
            self._publish(slot, payload, recv_ns)

    def _release(self, slot: _Slot) -> None:
        """Hand a published slot back, re-queueing it when a newer payload landed meanwhile."""
        if not slot.payload:
            slot.owned = False
            return
        self._pending.append(slot)
        self._cond.notify()

    def _publish(self, slot: _Slot, payload: bytes, recv_ns: int) -> None:
        """Worker thread: decode, convert and log one message to its foxglove channel."""
        try:
            msg = slot.decode(payload)
            log_ns = converters.stamp_ns(msg, recv_ns)
            if slot.convert is None:
                slot.channel.log(converters.to_json(msg), log_time=log_ns)
                return
            slot.channel.log(slot.convert(msg, converters.timestamp(log_ns)), log_time=log_ns)
        except Exception:
            if slot.warned:
                return
            slot.warned = True
            logger.warning(
                "foxglove bridge cannot serve topic", topic=slot.channel.topic(), exc_info=True
            )

    def _stop_workers(self) -> None:
        """Wake every worker so it observes the stop flag, then join it."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for worker in self._workers:
            worker.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._workers.clear()

    def _close_slots(self) -> None:
        """Unadvertise every channel and report the topics that dropped frames."""
        with self._cond:  # the bus thread is still subscribed and still inserting slots
            slots = {n: s for n, s in self._slots.items() if s is not None}
            self._slots.clear()
            self._pending.clear()
        dropped = {n: s.dropped for n, s in slots.items() if s.dropped}
        for slot in slots.values():
            slot.channel.close()
        logger.info("foxglove bridge stopped", dropped=dropped)


def run_bridge(host: str = FOXGLOVE_HOST, port: int = FOXGLOVE_PORT) -> None:
    """Start a FoxgloveBridgeModule on the running dimos bus and block until interrupted."""
    autoconf(check_only=True)

    bridge = FoxgloveBridgeModule(host=host, port=port)
    # standalone has no coordinator, so the teleop Outs need the topics a blueprint would assign
    bridge.tele_cmd_vel.transport = make_transport(f"/{_TELE_CMD_VEL}", Twist)
    bridge.clicked_point.transport = make_transport(f"/{_CLICKED_POINT}", PointStamped)
    bridge.start()

    def _shutdown(*_: object) -> None:
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.pause()
