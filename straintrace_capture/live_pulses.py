"""Bounded live receipt and ledger counts for explicitly anchored hardware pulses."""

from dataclasses import dataclass
from heapq import heappush, heapreplace


RECENT_PULSE_GROUPS = 24


@dataclass(slots=True)
class _Camera:
    highest: int | None = None
    latest: int | None = None
    unidentified: int = 0
    backtracks: int = 0


@dataclass(slots=True)
class _Receipts:
    received: int = 0
    saved: int = 0


class LivePulseGroups:
    """Call under the session lock. Keep the planned start and the latest observed numbers."""

    def __init__(self, phase, pairing):
        self.phase, self.pairing = phase, pairing
        self.first = pairing.first_event_number
        self.cameras = {camera: _Camera() for camera in pairing.first_counter}
        self._recent = []
        self._groups = {self.first: {}}

    def receive(self, camera, number):
        observation = self.cameras[camera]
        if number is None:
            observation.unidentified += 1
            return
        if observation.latest is not None and number < observation.latest:
            observation.backtracks += 1
        observation.latest = number
        if observation.highest is None or number > observation.highest:
            observation.highest = number
        group = self._groups.get(number)
        if group is None:
            if len(self._recent) < RECENT_PULSE_GROUPS:
                heappush(self._recent, number)
            elif number > self._recent[0]:
                del self._groups[heapreplace(self._recent, number)]
            else:
                return
            group = self._groups[number] = {}
        counts = group.get(camera)
        if counts is None:
            counts = group[camera] = _Receipts()
        counts.received += 1

    def commit(self, camera, number):
        if (group := self._groups.get(number)) is not None and (counts := group.get(camera)) is not None:
            counts.saved += 1

    def snapshot(self, *, accepting):
        cameras = {
            camera: {"first_counter": self.pairing.first_counter[camera], "highest_number": observation.highest,
                     "latest_number": observation.latest, "unidentified": observation.unidentified,
                     "backtracks": observation.backtracks}
            for camera, observation in self.cameras.items()
        }
        groups = []
        for number in (self.first, *sorted(self._recent, reverse=True)):
            members = self._groups[number]
            advanced = [camera for camera, row in cameras.items()
                        if camera not in members and row["highest_number"] is not None and row["highest_number"] > number]
            duplicates = [camera for camera, counts in members.items() if counts.received > 1]
            groups.append({
                "number": number, "received_cameras": len(members),
                "saved_cameras": sum(counts.saved > 0 for counts in members.values()),
                "advanced": advanced, "duplicates": duplicates,
                "cameras": {camera: {"received": counts.received, "saved": counts.saved} for camera, counts in members.items()},
            })
        return {"phase": self.phase, "source_id": self.pairing.source_id, "counter_feature": self.pairing.counter_feature,
                "first_number": self.first, "accepting": accepting, "recent_limit": RECENT_PULSE_GROUPS,
                "cameras": cameras, "groups": groups}
