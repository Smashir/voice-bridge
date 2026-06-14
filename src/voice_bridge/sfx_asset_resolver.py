"""Provider-independent SFX asset resolver for voice-bridge.

This module reads one or more normalized SFX records.jsonl files and resolves
a natural-language cue/prompt into candidate audio assets.

It is intentionally provider-independent:
- provider-specific source metadata stays in record["source_classification"]
- runtime control tags stay in record["tags"]
- this resolver uses only the common records.jsonl schema

No files are written by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RECORDS_ENV = "VOICE_BRIDGE_SFX_RECORDS"
TAG_KEYS = ["event", "material", "source", "surface", "scene", "usage", "acoustic"]


@dataclass(frozen=True)
class SfxCandidate:
    score: float
    record: dict[str, Any]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        rec = self.record
        return {
            "score": self.score,
            "id": rec.get("id"),
            "provider": rec.get("provider"),
            "path": rec.get("path"),
            "title": rec.get("title"),
            "labels": labels_text(rec),
            "tags": rec.get("tags") or {},
            "composition": rec.get("composition"),
            "continuity": rec.get("continuity"),
            "asset_duration_sec": rec.get("asset_duration_sec"),
            "asset_count_hint": rec.get("asset_count_hint"),
            "reasons": self.reasons,
        }


def has_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def add(xs: list[str], value: str) -> None:
    if value and value not in xs:
        xs.append(value)


def parse_int_before_units(text: str, units: list[str]) -> int | None:
    unit_pat = "|".join(re.escape(u) for u in units)
    m = re.search(rf"(\d+)\s*({unit_pat})", text)
    if m:
        return int(m.group(1))

    kanji = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    for k, v in kanji.items():
        for u in units:
            if f"{k}{u}" in text:
                return v

    return None


def records_paths_from_env(env_value: str | None = None) -> list[Path]:
    raw = env_value if env_value is not None else os.getenv(RECORDS_ENV, "")
    raw = (raw or "").strip()
    if not raw:
        return []

    # Linux path list uses ":"; also accept comma/newline for convenience.
    parts: list[str] = []
    for chunk in raw.replace("\n", ":").replace(",", ":").split(":"):
        p = chunk.strip()
        if p:
            parts.append(p)

    return [Path(p) for p in parts]


def load_records(records_paths: list[str | Path] | None = None) -> list[dict[str, Any]]:
    paths = [Path(p) for p in records_paths] if records_paths else records_paths_from_env()
    if not paths:
        return []

    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"[sfx-resolver] WARN: records file not found: {path}")
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(rec, dict):
                    out.append(rec)

    return out


def labels_text(record: dict[str, Any]) -> str:
    labels: list[str] = []
    for item in record.get("source_classification") or []:
        if isinstance(item, dict):
            label = item.get("label")
            if isinstance(label, str) and label:
                labels.append(label)
    return " ".join(labels)


def record_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("title") or "",
        record.get("note") or "",
        record.get("raw_text") or "",
        record.get("embedding_text") or "",
        labels_text(record),
        record.get("filename") or "",
        record.get("path") or "",
    ]
    return " ".join(str(p) for p in parts if p)


def tagset(record: dict[str, Any], key: str) -> set[str]:
    tags = record.get("tags") or {}
    values = tags.get(key) or []
    return set(str(v) for v in values if v)


def infer_profile(query: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    q = (query or "").strip()
    constraints = constraints or {}

    profile: dict[str, Any] = {
        "events": [],
        "materials": [],
        "sources": [],
        "surfaces": [],
        "scenes": [],
        "usages": [],
        "acoustics": [],
        "avoid_events": [],
        "avoid_acoustics": ["poor_recording", "clipped"],
        "prefer_composition": None,
        "allow_mixed": False,
        "prefer_continuity": None,
        "target_count": None,
        "target_duration_sec": None,
        "keywords": [],
    }

    count = parse_int_before_units(q, ["回", "歩", "発"])
    if count is not None:
        profile["target_count"] = count

    duration = parse_int_before_units(q, ["秒"])
    if duration is not None:
        profile["target_duration_sec"] = float(duration)

    # Scene
    if has_any(q, ["神社"]):
        add(profile["scenes"], "shrine")
    if has_any(q, ["寺", "お寺"]):
        add(profile["scenes"], "temple")
    if has_any(q, ["学校", "教室"]):
        add(profile["scenes"], "school")
    if has_any(q, ["駅", "ホーム"]):
        add(profile["scenes"], "station")
    if has_any(q, ["住宅街"]):
        add(profile["scenes"], "residential")
    if has_any(q, ["森", "山"]):
        add(profile["scenes"], "forest")
    if has_any(q, ["公園"]):
        add(profile["scenes"], "park")
    if has_any(q, ["駐車場"]):
        add(profile["scenes"], "parking_lot")

    # Weather / ambience
    if has_any(q, ["雨", "雨音", "大雨", "雷雨"]):
        add(profile["events"], "rain")
        add(profile["usages"], "ambience")
        profile["prefer_continuity"] = "continuous"

    if has_any(q, ["風", "強風", "嵐", "台風", "暴風"]):
        add(profile["events"], "wind")
        add(profile["usages"], "ambience")
        profile["prefer_continuity"] = "continuous"

    if has_any(q, ["雷", "雷鳴", "雷雨"]):
        add(profile["events"], "thunder")
        add(profile["usages"], "impact")

    weather_events = set(profile["events"]) & {"rain", "wind", "thunder"}
    if len(weather_events) >= 2 or has_any(q, ["重ね", "同時", "混ぜ", "混ざ", "中で"]):
        profile["allow_mixed"] = True

    # Animals
    if has_any(q, ["鳥", "小鳥", "スズメ", "カラス", "ウグイス", "鳩"]):
        add(profile["events"], "bird_chirp")
        add(profile["sources"], "bird")
        add(profile["sources"], "animal")
        add(profile["usages"], "animal_voice")

    if has_any(q, ["セミ", "蝉", "虫", "コオロギ", "鈴虫"]):
        add(profile["events"], "insect_chirp")
        add(profile["sources"], "insect")
        add(profile["sources"], "animal")
        add(profile["usages"], "animal_voice")

    # Footsteps
    if has_any(q, ["足音", "歩", "歩く", "走る", "階段"]):
        add(profile["events"], "footsteps")
        add(profile["sources"], "human")
        add(profile["usages"], "foley")
        add(profile["usages"], "walking_sequence")
        profile["prefer_continuity"] = "sequence"
        if not profile["allow_mixed"]:
            profile["prefer_composition"] = "isolated"

    if has_any(q, ["コンクリート"]):
        add(profile["surfaces"], "concrete")
    if has_any(q, ["砂利"]):
        add(profile["surfaces"], "gravel")
    if has_any(q, ["落ち葉", "枯葉"]):
        add(profile["surfaces"], "leaves")
    if has_any(q, ["草"]):
        add(profile["surfaces"], "grass")
    if has_any(q, ["土", "地面"]):
        add(profile["surfaces"], "soil")
    if has_any(q, ["タイル"]):
        add(profile["surfaces"], "tile")

    # Doors / knocks / props
    if has_any(q, ["ノック", "叩く", "コンコン", "トントン"]):
        add(profile["events"], "knock")
        add(profile["usages"], "foley")
        if profile["target_count"]:
            profile["prefer_continuity"] = "sequence"
        if profile["prefer_composition"] is None and not profile["allow_mixed"]:
            profile["prefer_composition"] = "isolated"
        add(profile["avoid_events"], "pen_click")
        add(profile["avoid_events"], "vehicle_door")

    if has_any(q, ["ペン", "ボールペン", "シャーペン"]):
        add(profile["events"], "pen_click")
        add(profile["sources"], "pen")
        add(profile["usages"], "prop_foley")

    if has_any(q, ["ドア", "扉", "引き戸"]):
        add(profile["events"], "door")
        add(profile["usages"], "foley")

    if has_any(q, ["戸棚", "棚", "引き出し", "下駄箱", "クローゼット", "ロッカー"]):
        add(profile["events"], "cabinet_door")
        add(profile["usages"], "prop_foley")

    if has_any(q, ["電車ドア", "車両ドア"]):
        add(profile["events"], "vehicle_door")
        add(profile["sources"], "train")
        add(profile["sources"], "vehicle")

    # Materials / breakage
    if has_any(q, ["木", "木製"]):
        add(profile["materials"], "wood")

    if has_any(q, ["金属", "鉄"]):
        add(profile["materials"], "metal")

    if has_any(q, ["ガラス", "ビン", "瓶", "パリン", "ガシャン"]):
        add(profile["events"], "glass_break")
        add(profile["materials"], "glass")
        add(profile["usages"], "impact")
        add(profile["avoid_events"], "egg_crack")
        add(profile["avoid_events"], "wet_crack")
        if profile["prefer_composition"] is None and not profile["allow_mixed"]:
            profile["prefer_composition"] = "isolated"

    if has_any(q, ["卵", "玉子"]):
        add(profile["events"], "egg_crack")
        add(profile["events"], "wet_crack")
        add(profile["sources"], "egg")
        add(profile["materials"], "shell")
        add(profile["materials"], "organic")
        add(profile["materials"], "liquid")
        add(profile["usages"], "foley")
        add(profile["avoid_events"], "glass_break")
        if profile["prefer_composition"] is None and not profile["allow_mixed"]:
            profile["prefer_composition"] = "isolated"

    if has_any(q, ["水", "波", "川"]):
        add(profile["events"], "water")
        add(profile["materials"], "water")

    # Useful text keywords
    for word in [
        "木製",
        "硬い木",
        "薄い木",
        "教室",
        "神社",
        "寺",
        "住宅街",
        "マンション",
        "コンクリート",
        "砂利",
        "落ち葉",
        "階段",
        "地面",
        "床",
        "生卵",
        "卵",
        "ガラス",
        "ビン",
        "雨",
        "雷",
        "強風",
        "足音",
    ]:
        if word in q:
            add(profile["keywords"], word)

    # External constraints from render_plan/audio_plan can override the inferred profile.
    for key in [
        "events",
        "materials",
        "sources",
        "surfaces",
        "scenes",
        "usages",
        "acoustics",
        "avoid_events",
        "avoid_acoustics",
        "keywords",
    ]:
        values = constraints.get(key)
        if isinstance(values, list):
            for value in values:
                add(profile[key], str(value))

    for key in ["prefer_composition", "prefer_continuity"]:
        value = constraints.get(key)
        if isinstance(value, str) and value:
            profile[key] = value

    if isinstance(constraints.get("allow_mixed"), bool):
        profile["allow_mixed"] = bool(constraints["allow_mixed"])

    if isinstance(constraints.get("target_count"), int):
        profile["target_count"] = int(constraints["target_count"])

    if isinstance(constraints.get("target_duration_sec"), (int, float)):
        profile["target_duration_sec"] = float(constraints["target_duration_sec"])

    return profile


def duration_score(record: dict[str, Any], profile: dict[str, Any]) -> float:
    dur = record.get("asset_duration_sec")
    if not isinstance(dur, (int, float)):
        return -2.0

    events = tagset(record, "event")
    target_count = profile.get("target_count")
    target_duration = profile.get("target_duration_sec")

    if target_duration:
        if dur >= target_duration:
            return 12.0
        return -min(20.0, (target_duration - dur) * 0.8)

    if target_count and "footsteps" in events:
        expected = target_count * 0.45
        diff = abs(float(dur) - expected)
        return max(-8.0, 12.0 - diff * 1.5)

    if target_count and "knock" in events:
        if 0.25 <= float(dur) <= max(8.0, target_count * 2.0):
            return 10.0
        return -4.0

    return 0.0


def count_hint_score(record: dict[str, Any], profile: dict[str, Any]) -> tuple[float, str | None]:
    target = profile.get("target_count")
    if not target:
        return 0.0, None

    hint = record.get("asset_count_hint")
    if not isinstance(hint, int):
        return 0.0, None

    diff = abs(hint - target)

    if diff == 0:
        return 18.0, f"+18 count:{hint}"

    if diff == 1:
        return 4.0, f"+4 near count:{hint}"

    return -10.0, f"-10 count mismatch:{hint}"


def score_record(record: dict[str, Any], profile: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    events = tagset(record, "event")
    materials = tagset(record, "material")
    sources = tagset(record, "source")
    surfaces = tagset(record, "surface")
    scenes = tagset(record, "scene")
    usages = tagset(record, "usage")
    acoustics = tagset(record, "acoustic")
    text = record_text(record)

    def match_values(label: str, desired: list[str], actual: set[str], weight: float) -> None:
        nonlocal score
        for x in desired:
            if x in actual:
                score += weight
                reasons.append(f"+{int(weight)} {label}:{x}")

    match_values("event", profile["events"], events, 80)
    match_values("material", profile["materials"], materials, 30)
    match_values("source", profile["sources"], sources, 25)
    match_values("surface", profile["surfaces"], surfaces, 30)
    match_values("scene", profile["scenes"], scenes, 25)
    match_values("usage", profile["usages"], usages, 15)
    match_values("acoustic", profile["acoustics"], acoustics, 10)

    if profile["events"]:
        matched_events = set(profile["events"]) & events
        if not matched_events:
            score -= 35
            reasons.append("-35 no desired event")

    for x in profile["avoid_events"]:
        if x in events:
            score -= 100
            reasons.append(f"-100 avoid event:{x}")

    for x in profile["avoid_acoustics"]:
        if x in acoustics:
            score -= 18
            reasons.append(f"-18 quality:{x}")

    if "noisy" in acoustics:
        score -= 8
        reasons.append("-8 noisy")
    if "wind_noise" in acoustics:
        score -= 8
        reasons.append("-8 wind_noise")
    if "raw_recording" in acoustics:
        score -= 3
        reasons.append("-3 raw_recording")

    composition = record.get("composition") or "unknown"
    prefer_comp = profile.get("prefer_composition")

    if prefer_comp == "isolated":
        if composition == "isolated":
            score += 12
            reasons.append("+12 isolated")
        elif composition == "mixed":
            if profile.get("allow_mixed"):
                score -= 2
                reasons.append("-2 mixed allowed")
            else:
                score -= 25
                reasons.append("-25 mixed")
        else:
            score -= 3
            reasons.append("-3 unknown composition")

    elif prefer_comp == "mixed":
        if composition == "mixed":
            score += 18
            reasons.append("+18 mixed")
        elif composition == "isolated":
            score += 2
            reasons.append("+2 isolated usable as layer")

    if profile.get("prefer_continuity"):
        cont = record.get("continuity")
        if cont == profile["prefer_continuity"]:
            score += 10
            reasons.append(f"+10 continuity:{cont}")

    ds = duration_score(record, profile)
    if ds:
        score += ds
        reasons.append(f"{ds:+.1f} duration")

    cs, cr = count_hint_score(record, profile)
    if cs:
        score += cs
    if cr:
        reasons.append(cr)

    for kw in profile["keywords"]:
        if kw in text:
            score += 8
            reasons.append(f"+8 kw:{kw}")

    label_text = labels_text(record)
    for kw in profile["keywords"]:
        if kw in label_text:
            score += 5
            reasons.append(f"+5 label:{kw}")

    return score, reasons


def resolve_sfx(
    query: str,
    *,
    top: int = 5,
    records: list[dict[str, Any]] | None = None,
    records_paths: list[str | Path] | None = None,
    constraints: dict[str, Any] | None = None,
    min_score: float = 0.0,
) -> list[SfxCandidate]:
    loaded = records if records is not None else load_records(records_paths)
    profile = infer_profile(query, constraints=constraints)

    candidates: list[SfxCandidate] = []
    for rec in loaded:
        score, reasons = score_record(rec, profile)
        if score > min_score:
            candidates.append(SfxCandidate(score=score, record=rec, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: max(0, int(top))]


def resolve_segment(
    segment: dict[str, Any],
    *,
    top: int = 5,
    records: list[dict[str, Any]] | None = None,
    records_paths: list[str | Path] | None = None,
) -> list[SfxCandidate]:
    query = str(
        segment.get("cue")
        or segment.get("prompt")
        or segment.get("display_text")
        or segment.get("text")
        or ""
    ).strip()

    constraints: dict[str, Any] = {}

    seg_type = str(segment.get("type") or segment.get("kind") or "").lower()
    if seg_type == "ambience":
        constraints["usages"] = ["ambience"]
        constraints["prefer_continuity"] = "continuous"
        constraints["allow_mixed"] = True
    elif seg_type == "foley":
        constraints["usages"] = ["foley"]
    elif seg_type == "music":
        constraints["usages"] = ["music"]

    level_db = segment.get("level_db")
    if isinstance(level_db, (int, float)):
        constraints["level_db"] = float(level_db)

    return resolve_sfx(
        query,
        top=top,
        records=records,
        records_paths=records_paths,
        constraints=constraints,
    )


def format_tags(record: dict[str, Any]) -> str:
    tags = record.get("tags") or {}
    parts: list[str] = []
    for key in TAG_KEYS:
        values = tags.get(key) or []
        if values:
            parts.append(f"{key}={','.join(str(v) for v in values)}")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--records", action="append", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-path", action="store_true")
    args = ap.parse_args()

    records_paths = args.records if args.records else None
    records = load_records(records_paths)
    results = resolve_sfx(args.query, top=args.top, records=records)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return

    print("query:", args.query)
    print("records:", len(records))
    print("matched:", len(results))
    print()

    for i, cand in enumerate(results, start=1):
        rec = cand.record
        dur = rec.get("asset_duration_sec")
        dur_s = f"{dur:.3f}s" if isinstance(dur, (int, float)) else "null"

        print(f"#{i} score={cand.score:.1f} id={rec.get('id')}")
        print("title:", rec.get("title") or "")
        print("labels:", labels_text(rec))
        print("tags:", format_tags(rec))
        print(
            "composition:",
            rec.get("composition"),
            "continuity:",
            rec.get("continuity"),
            "duration:",
            dur_s,
        )
        print("reasons:", "; ".join(cand.reasons[:12]))
        if args.show_path:
            print("path:", rec.get("path"))
        print()


if __name__ == "__main__":
    main()