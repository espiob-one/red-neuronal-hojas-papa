"""Isolated, conservative association prototype; not installed in the rover.

Adds globally optimal bounded matching, constant-velocity prediction and optional
RGB descriptors to LeafTracker. Existing token, expiry and reset mechanics are
inherited. Ambiguous/occluded observations invalidate cached classifications and
pending tokens; labels resume only after independent clear observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from leaf_tracks import LeafTracker, intersection_over_union


def optimal_assignment(costs, unmatched_cost=.65):
    """Return detection->track-index assignment and total global cost.

    A row is an existing track; a column is a detection. Both sides may remain
    unmatched. Bitmask dynamic programming is bounded at eight per side.
    """
    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    if max(rows, columns) > 8 or any(len(row) != columns for row in costs):
        raise ValueError("Assignment must be rectangular and no larger than 8x8")

    @lru_cache(None)
    def solve(row, used):
        if row == rows:
            return unmatched_cost * (columns - used.bit_count()), ()
        cost, tail = solve(row + 1, used)
        best = (cost + unmatched_cost, (-1,) + tail)
        for column in range(columns):
            if used & (1 << column) or not math.isfinite(costs[row][column]):
                continue
            cost, tail = solve(row + 1, used | (1 << column))
            option = (cost + costs[row][column], (column,) + tail)
            if option[0] < best[0]:
                best = option
        return best

    total, selected = solve(0, 0)
    return {column: row for row, column in enumerate(selected) if column >= 0}, total


def appearance_descriptor(image, box):
    """Small spatial RGB descriptor; no disease/identity labels are consumed."""
    from PIL import Image
    bounds = [round(float(value)) for value in box]
    crop = image.crop(bounds).convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
    return (np.asarray(crop, dtype=np.float32).reshape(-1) / 255).tolist()


def descriptor(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if 0 < len(array) <= 768 and np.isfinite(array).all():
            return array
    except (ValueError, TypeError):
        pass
    return None


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class MotionLeafTracker(LeafTracker):
    """LeafTracker-compatible prototype with at most eight active tracks.

    Optional detection field ``appearance`` is a finite descriptor vector.
    ``association_ambiguous`` and ``identity_safe`` explain abstention. Descriptor
    anchors and velocities are updated only on clean, isolated observations.
    """

    def __init__(self, *args, ambiguity_margin=.12, overlap_abstention=.12,
                 appearance_change=.20, clear_hits=2, **kwargs):
        if len(args) < 5:
            kwargs.setdefault("max_tracks", 8)
        super().__init__(*args, **kwargs)
        if (self.max_tracks > 8 or clear_hits < 1 or not 0 <= ambiguity_margin <= 1
                or not 0 <= overlap_abstention <= 1 or not 0 < appearance_change <= 1):
            raise ValueError("Prototype supports at most eight tracks and positive clear_hits")
        self.ambiguity_margin = ambiguity_margin
        self.overlap_abstention = overlap_abstention
        self.appearance_change = appearance_change
        self.clear_hits = clear_hits

    def _prediction(self, track):
        anchor = track.get("_anchor_box", track["box"])
        age = self._now - track.get("_anchor_time", track["last_seen"])
        vx, vy = track.get("_velocity", (0., 0.))
        return (anchor[0] + vx * age, anchor[1] + vy * age,
                anchor[2] + vx * age, anchor[3] + vy * age)

    def _cost(self, track, item):
        predicted, box = self._prediction(track), item["box"]
        px, py = center(predicted)
        dx, dy = center(box)
        diagonal = max(1e-9, math.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
        distance = math.hypot(px - dx, py - dy) / diagonal
        iou = intersection_over_union(predicted, box)
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1]) /
                      ((predicted[2] - predicted[0]) * (predicted[3] - predicted[1])))
        if (iou < .05 and distance > .65) or not .3 <= area_ratio <= 3:
            return math.inf
        prior, current = track.get("_appearance"), item["appearance"]
        appearance_cost = 0.
        if prior is not None and current is not None and prior.shape == current.shape:
            appearance_cost = float(np.abs(prior - current).mean())
            if appearance_cost > self.appearance_change:
                return math.inf
        return .45 * (1 - iou) + .30 * min(distance, 1) + .25 * min(appearance_cost * 4, 1)

    def _row(self, track):
        row = super()._row(track)
        row = {key: value for key, value in row.items() if not key.startswith("_")}
        safe = (not track.get("association_ambiguous", False)
                and track.get("clear_observations", 0) >= self.clear_hits)
        row["identity_safe"] = bool(safe)
        if not safe or not row["observed"] or not track["quality_ok"]:
            row.update(result=None, ready=False)
        return row

    def update(self, detections, timestamp, image_signature=None):
        valid = []
        for index, item in enumerate(detections):
            try:
                box = tuple(float(value) for value in item["box"])
                score = float(item["score"])
                if (len(box) != 4 or not all(math.isfinite(v) for v in (*box, score))
                        or box[2] <= box[0] or box[3] <= box[1]):
                    continue
                valid.append({"detection_index": index, "box": box, "score": score,
                              "quality_ok": bool(item.get("ready", False)),
                              "appearance": descriptor(item.get("appearance"))})
            except (KeyError, TypeError, ValueError):
                continue
        valid = sorted(valid, key=lambda item: -item["score"])[:8]
        self._advance(timestamp)
        ids = sorted(self._tracks)
        costs = [[self._cost(self._tracks[tid], item) for item in valid] for tid in ids]
        # A detector can merge two occluding leaves into ONE box. Appearance may
        # exclude one candidate, but that does not prove that the box is a single
        # object. Preserve this geometric ambiguity before descriptor gating.
        support = []
        for item in valid:
            count = 0
            for tid in ids:
                predicted = self._prediction(self._tracks[tid])
                cx, cy = center(predicted)
                box = item["box"]
                if (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]) or (
                        intersection_over_union(predicted, box) >= self.iou_threshold):
                    count += 1
            support.append(count)
        chosen, _ = optimal_assignment(costs)
        assignments = {di: ids[ti] for di, ti in chosen.items()}
        used = set(assignments.values())
        # A missed observation cannot publish or complete its previous result.
        for tid in self._tracks:
            if tid not in used:
                self.invalidate_classification(tid)
                self._tracks[tid]["clear_observations"] = 0
        for di in range(len(valid)):
            if di in assignments or len(self._tracks) >= self.max_tracks:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"track_id": tid, "first_seen": self._now,
                                 "hits": 0, "result": None, "_token": None,
                                 "clear_observations": 0}
            assignments[di] = tid
        self._observed_ids = set(assignments.values())
        rows = []
        for di, tid in sorted(assignments.items(), key=lambda pair: valid[pair[0]]["detection_index"]):
            item, track = valid[di], self._tracks[tid]
            ti = ids.index(tid) if tid in ids else None
            competitors = []
            if ti is not None:
                competitors.extend(costs[ti][j] - costs[ti][di] for j in range(len(valid)) if j != di)
                competitors.extend(costs[j][di] - costs[ti][di] for j in range(len(ids)) if j != ti)
            cost_ambiguous = bool(competitors and min(competitors) < self.ambiguity_margin)
            overlapping = any(intersection_over_union(item["box"], other["box"]) > self.overlap_abstention
                              for j, other in enumerate(valid) if j != di)
            ambiguous = bool(cost_ambiguous or overlapping or support[di] > 1)
            new_observation = track.get("last_seen") != self._now
            track["hits"] += int(new_observation)
            clear = item["quality_ok"] and not ambiguous
            track["clear_observations"] = (track["clear_observations"] + int(new_observation)) if clear else 0
            if not clear:
                self.invalidate_classification(tid)
            if clear:
                prior_box, prior_time = track.get("_anchor_box"), track.get("_anchor_time")
                if prior_box is not None and self._now > prior_time:
                    before, after = center(prior_box), center(item["box"])
                    velocity = ((after[0] - before[0]) / (self._now - prior_time),
                                (after[1] - before[1]) / (self._now - prior_time))
                    previous = track.get("_velocity", velocity)
                    track["_velocity"] = tuple(.8 * v + .2 * p for v, p in zip(velocity, previous))
                track.update(_anchor_box=item["box"], _anchor_time=self._now)
                if item["appearance"] is not None:
                    track["_appearance"] = item["appearance"].copy()
            track.update(box=item["box"], score=item["score"], quality_ok=item["quality_ok"],
                         last_seen=self._now, image_signature=image_signature,
                         detection_index=item["detection_index"], association_ambiguous=ambiguous,
                         supporting_prediction_count=support[di],
                         association_cost=costs[ti][di] if ti is not None else None)
            track["confirmed"] = (track["hits"] >= self.min_hits and self._now - track["first_seen"] >= self.min_age)
            rows.append(self._row(track))
        return rows

    def finish_classification(self, token, result, timestamp):
        self._advance(timestamp)
        if not isinstance(token, tuple) or len(token) != 3:
            return False
        track = self._tracks.get(token[0])
        if (track is None or token[0] not in self._observed_ids or not track["quality_ok"]
                or track.get("association_ambiguous")
                or track.get("clear_observations", 0) < self.clear_hits):
            return False
        return super().finish_classification(token, result, timestamp)


def evaluate_replay(source):
    """Compare frozen 120-frame detections; annotations never enter matching."""
    from evaluate_robustness_suite import Classifier, digest, leaf_asset, render_video_frame, write_json
    from rover_motion import leaf_crop
    source_metrics = json.loads((source / "replay_metrics.json").read_text())
    negative_rows = json.loads((source / "negative_rows_snapshot.json").read_text())
    source_frozen = Path(json.loads((source / "summary.json").read_text())["source_frozen_report"])
    config_path = source_frozen / "candidate/config.json"
    classifier = Classifier(source_frozen / "candidate/model.onnx", config_path, 2)
    classes = classifier.config["classes"]
    sources = [(row, leaf_asset(row)[1]) for row in source_metrics["source_rows"]]
    background_path = source_metrics["negative_background"]
    if background_path not in {row["path"] for row in negative_rows}:
        raise ValueError("Replay background not in reviewed negative snapshot")
    with Image.open(Path(__file__).parent / background_path) as image:
        background = image.convert("RGB")
    output = Path(__file__).parent / "reports" / ("tracking_candidate_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    variants = {"iou_baseline": LeafTracker(), "motion_only": MotionLeafTracker(),
                "motion_appearance": MotionLeafTracker()}
    records = {name: [] for name in variants}
    counters = {name: Counter() for name in variants}
    observations = {name: defaultdict(list) for name in variants}
    calls = {name: [] for name in variants}
    runtimes, descriptor_runtimes = defaultdict(list), []
    ffmpeg = shutil.which("ffmpeg")
    encoder = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "1280x360", "-r", "8", "-i", "pipe:0", "-an", "-c:v", "libx264",
                                "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", str(output / "comparison_120frames.mp4")],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE) if ffmpeg else None
    try:
        for saved in source_metrics["frames"]:
            index, timestamp = saved["frame"], saved["simulated_timestamp"]
            image, truth = render_video_frame(index, sources, background)
            if truth != saved["ground_truth"]:
                raise RuntimeError("Replay regeneration differs from frozen ground truth")
            detections = saved["detections"]
            started = time.perf_counter()
            descriptors = [appearance_descriptor(image, detection["box"]) for detection in detections]
            descriptor_runtimes.append((time.perf_counter() - started) * 1000)
            matched, used = {}, set()
            pairs = sorted((-intersection_over_union(d["box"], g["box"]), di, gi)
                           for di, d in enumerate(detections) for gi, g in enumerate(truth))
            for negative_iou, di, gi in pairs:
                if -negative_iou < .3:
                    break
                if di not in matched and gi not in used:
                    matched[di] = gi
                    used.add(gi)
            for name, tracker in variants.items():
                inputs = [{**d, **({"appearance": descriptors[i]} if name == "motion_appearance" else {})}
                          for i, d in enumerate(detections)]
                started = time.perf_counter()
                tracks = tracker.update(inputs, timestamp)
                runtimes[name].append((time.perf_counter() - started) * 1000)
                rows = []
                for track in tracks:
                    target = truth[matched[track["detection_index"]]] if track["detection_index"] in matched else None
                    if target:
                        observations[name][target["object_id"]].append((index, track["track_id"]))
                    if track["ready"]:
                        token = tracker.begin_classification(track["track_id"], track["last_seen"])
                        if token:
                            # Truth is retained only as evaluation provenance of
                            # this crop; the tracker receives no ground truth.
                            crop = leaf_crop(image, track["box"])
                            scores = classifier.predict(crop)
                            result = {"source_object_id": target["object_id"] if target else None,
                                      "class": classes[int(scores.argmax())], "score": float(scores.max())}
                            accepted = tracker.finish_classification(token, result, timestamp + .0001)
                            calls[name].append({"frame": index, "track_id": track["track_id"],
                                                "accepted": accepted, "result": result})
                    snapshot_row = next(t for t in tracker.snapshot()["tracks"] if t["track_id"] == track["track_id"])
                    result = snapshot_row["result"]
                    wrong = bool(result and target and result["source_object_id"] != target["object_id"])
                    if target:
                        counters[name]["matched_observations"] += 1
                        counters[name]["wrong_source_result_observations"] += wrong
                        counters[name]["abstained_observations"] += result is None
                        counters[name]["correct_source_result_observations"] += bool(result and not wrong)
                    counters[name]["identity_ambiguous_observations"] += bool(track.get("association_ambiguous"))
                    rows.append({"track_id": track["track_id"], "box": track["box"],
                                 "target": target["object_id"] if target else None, "result": result,
                                 "wrong_source": wrong, "identity_safe": track.get("identity_safe"),
                                 "supporting_prediction_count": track.get("supporting_prediction_count"),
                                 "ambiguous": track.get("association_ambiguous", False)})
                records[name].append({"frame": index, "tracks": rows})
            comparison = Image.new("RGB", (1280, 360))
            for panel, name in enumerate(("iou_baseline", "motion_appearance")):
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for track in records[name][-1]["tracks"]:
                    color = "red" if track["wrong_source"] else "yellow" if track["result"] is None else "lime"
                    draw.rectangle(track["box"], outline=color, width=2)
                    text = f"ID{track['track_id']} " + (track["result"]["class"].split("___")[-1] if track["result"] else "ABSTAIN")
                    draw.text((max(0, track["box"][0]), max(20, track["box"][1] - 15)), text,
                              fill=color, stroke_width=1, stroke_fill="black")
                draw.text((7, 7), f"{name} frame {index + 1}/120", fill="white", stroke_width=1, stroke_fill="black")
                comparison.paste(annotated, (panel * 640, 0))
            if encoder:
                encoder.stdin.write(comparison.tobytes())
            if index in (48, 56, 64, 80, 104):
                comparison.save(output / f"comparison_{index:03}.jpg", quality=92)
    finally:
        if encoder:
            encoder.stdin.close()
            error = encoder.stderr.read().decode(errors="replace")
            if encoder.wait() != 0:
                raise RuntimeError(error)
    summaries = {}
    for name in variants:
        switches = {obj: sum(a[1] != b[1] and b[0] - a[0] <= 10 for a, b in zip(items, items[1:]))
                    for obj, items in observations[name].items()}
        summaries[name] = {"counts": dict(counters[name]), "identity_switches_per_object": switches,
                           "track_ids_per_object": {obj: sorted({item[1] for item in values})
                                                    for obj, values in observations[name].items()},
                           "classification_calls": calls[name], "classification_count": len(calls[name]),
                           "first_classification_frame": min((row["frame"] for row in calls[name]), default=None),
                           "tracker_update_ms_mean_pc": float(np.mean(runtimes[name])),
                           "tracker_update_ms_p95_pc": float(np.percentile(runtimes[name], 95)),
                           "empty_tracker_at_end": variants[name].snapshot()["counts"]["active"] == 0}
    write_json(output / "frames.json", records)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(output),
              "source_replay": str(source.resolve()), "source_replay_sha256": digest(source / "replay_metrics.json"),
              "prototype_sha256": digest(__file__), "frame_count": len(source_metrics["frames"]),
              "fixed_detector_totals": source_metrics["totals"], "variants": summaries,
              "descriptor_ms_mean_pc": float(np.mean(descriptor_runtimes)),
              "descriptor_ms_p95_pc": float(np.percentile(descriptor_runtimes, 95)),
              "parameters": {"ambiguity_margin": .12, "overlap_abstention": .12,
                             "appearance_change": .20, "clear_hits": 2, "max_tracks": 8},
              "development_note": "A merged-detection ambiguity guard was added after diagnosing the initial run on this replay; this is development validation, not an unseen tracking test.",
              "limits": ["One synthetic 120-frame replay; no user photos, field or hardware timing validation.",
                         "Detector observations are frozen and identical across variants; detector misses cannot improve.",
                         "No parameter or seed search; prototype default parameters were defined before this replay.",
                         "Abstention deliberately discards cached results and may require additional crop classifications.",
                         "The prototype is isolated from the live runtime and has not been deployed."]}
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-source", type=Path,
                        default=Path(__file__).parent / "reports/robustness_20261022T054453Z_negative_extension")
    evaluate_replay(parser.parse_args().replay_source)
