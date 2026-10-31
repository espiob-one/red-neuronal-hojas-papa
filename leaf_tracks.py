"""Small, model-independent leaf tracker. Coordinates may be pixels or normalized.

``update(detections, timestamp, image_signature=None)`` accepts dictionaries with
``box=[x1,y1,x2,y2]``, ``score`` and optional ``ready`` (default False). The caller
decides crop quality/size and owns the crops. Returned rows represent only this
frame's observations; ``detection_index`` identifies the corresponding crop.

``begin_classification(track_id, observation_timestamp)`` reserves one inference
and returns an opaque token, or None. ``finish_classification(token, result,
timestamp)`` returns whether the result was accepted. Refreshing the SAME track
while inference runs is allowed; an expired track or obsolete token is rejected.
IDs are never reused during a tracker lifetime. ``snapshot(timestamp=None)``
returns all active tracks and counts; passing a time also expires absent tracks.
``observed`` means present in the most recent detection update, not HTTP poll.
``reset(timestamp)`` invalidates all tracks and pending tokens without reusing IDs.
All timestamps must use the same monotonic clock, in seconds. No queue, model,
image processing or external package is used.
"""

import math


def intersection_over_union(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (
        b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.


class LeafTracker:
    def __init__(self, max_age=1.25, min_hits=2, min_age=0.15,
                 iou_threshold=0.25, max_tracks=16):
        if (max_age <= 0 or min_hits < 1 or min_age < 0 or
                not 0 < iou_threshold <= 1 or not 1 <= max_tracks <= 16):
            raise ValueError("Invalid tracker limits")
        self.max_age, self.min_hits, self.min_age = max_age, min_hits, min_age
        self.iou_threshold, self.max_tracks = iou_threshold, max_tracks
        self._tracks, self._next_id, self._request_id, self._now = {}, 1, 0, None
        self._observed_ids = set()

    def _advance(self, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self._now is not None and timestamp < self._now):
            raise ValueError("Timestamp must be finite and monotonic")
        self._now = timestamp
        self._tracks = {key: row for key, row in self._tracks.items()
                        if timestamp - row["last_seen"] <= self.max_age}

    def _row(self, track):
        row = {key: value for key, value in track.items() if key != "_token"}
        row["box"] = list(track["box"])
        row["observed"] = track["track_id"] in self._observed_ids
        row["pending"] = track["_token"] is not None
        row["ready"] = bool(track["quality_ok"] and track["confirmed"] and
                            row["observed"] and not row["pending"] and
                            track["result"] is None)
        return row

    def update(self, detections, timestamp, image_signature=None):
        valid = []
        for index, item in enumerate(detections):
            try:
                box = tuple(float(x) for x in item["box"])
                score = float(item["score"])
                if (len(box) != 4 or not all(math.isfinite(x) for x in (*box, score))
                        or box[2] <= box[0] or box[3] <= box[1]):
                    continue
                valid.append((index, box, score, bool(item.get("ready", False))))
            except (KeyError, TypeError, ValueError):
                continue
        self._advance(timestamp)
        # Global greedy matching prevents two detections claiming the same ID.
        pairs = sorted(((-intersection_over_union(t["box"], d[1]), tid, i)
                        for tid, t in self._tracks.items() for i, d in enumerate(valid)))
        assignments, used_tracks = {}, set()
        for negative_iou, tid, index in pairs:
            if -negative_iou < self.iou_threshold:
                break
            if tid not in used_tracks and index not in assignments:
                assignments[index] = tid
                used_tracks.add(tid)
        # With a crowded scene, strongest new candidates get the bounded slots.
        for index in sorted(range(len(valid)), key=lambda i: -valid[i][2]):
            if index in assignments or len(self._tracks) >= self.max_tracks:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"track_id": tid, "first_seen": self._now,
                                 "hits": 0, "result": None, "_token": None}
            assignments[index] = tid
        self._observed_ids = set(assignments.values())
        rows = []
        for index, tid in sorted(assignments.items()):
            detection_index, box, score, quality_ok = valid[index]
            track = self._tracks[tid]
            # Duplicate updates at one timestamp cannot confirm a new track.
            track["hits"] += int(track.get("last_seen") != self._now)
            track.update(box=box, score=score, quality_ok=quality_ok,
                         last_seen=self._now, image_signature=image_signature,
                         detection_index=detection_index)
            track["confirmed"] = (track["hits"] >= self.min_hits and
                                  self._now - track["first_seen"] >= self.min_age)
            rows.append(self._row(track))
        return rows

    def begin_classification(self, track_id, observation_timestamp):
        track = self._tracks.get(track_id)
        if (track is None or track["last_seen"] != observation_timestamp or
                not self._row(track)["ready"]):
            return None
        self._request_id += 1
        token = (track_id, self._request_id, observation_timestamp)
        track["_token"] = token
        return token

    def finish_classification(self, token, result, timestamp):
        self._advance(timestamp)
        if not isinstance(token, tuple) or len(token) != 3:
            return False
        track = self._tracks.get(token[0])
        if track is None or track["_token"] != token:
            return False
        track.update(result=result, _token=None)
        return True

    def cancel_classification(self, token):
        if not isinstance(token, tuple) or len(token) != 3:
            return False
        track = self._tracks.get(token[0])
        if track is None or track["_token"] != token:
            return False
        track["_token"] = None
        return True

    def invalidate_classification(self, track_id):
        """A materially changed crop must not retain the previous image's result."""
        track = self._tracks.get(track_id)
        if track is None:
            return False
        track.update(result=None, _token=None)
        return True

    def snapshot(self, timestamp=None):
        if timestamp is not None:
            self._advance(timestamp)
        tracks = [self._row(t) for t in self._tracks.values()]
        counts = {"active": len(tracks)}
        counts.update({key: sum(bool(t[key]) for t in tracks)
                       for key in ("observed", "confirmed", "ready", "pending")})
        counts["classified"] = sum(t["result"] is not None for t in tracks)
        return {"tracks": tracks, "counts": counts}

    def reset(self, timestamp):
        self._advance(timestamp)
        self._tracks.clear()
        self._observed_ids.clear()
