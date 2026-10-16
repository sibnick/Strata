#!/usr/bin/env python3
"""Create deterministic, natural-text retrieval prompts with the pack tokenizer."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PACK = Path(
    "/home/nikolay/projects/copilot-worktrees/170hx-3090/"
    "sibnick-upgraded-succotash/build/pack/ud-q4-k-xl"
)
TOKENIZER = PACK / "tokenizer"
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GENERATION_CAP = 192
CONTEXT = 262144
STAGES = (
    ("8k", 8192, 11),
    ("32k", 32768, 23),
    ("128k", 131072, 37),
    ("near-max", 261632, 53),
    ("near-max-repeat", 261632, 71),
)
EXPECTED = {
    "ORIGIN": "IRIS-4Q7-LANTERN",
    "CENTER": "CEDAR-8M2-QUARRY",
    "FINAL": "ORBIT-6V9-KETTLE",
}

PLACES = (
    "the north footbridge", "the orchard loading lane", "the east pump house",
    "the museum courtyard", "the lower ferry landing", "the old rail platform",
    "the market service road", "the hillside weather station", "the canal lock",
    "the west greenhouse", "the library annex", "the river workshop",
    "the coastal footpath", "the stone viaduct", "the community boat shed",
    "the hilltop water tank", "the farm access gate", "the south depot",
)
ROLES = ("a field coordinator", "a volunteer surveyor", "the evening custodian",
         "a maintenance technician", "the site steward", "a records clerk")
TOOLS = ("the handheld gauge", "the paper map", "the weather display",
         "the inspection ledger", "the radio check sheet", "the sample labels")
OBSERVATIONS = (
    "a narrow patch of shade along the retaining wall",
    "two delivery carts waiting beside the painted line",
    "a brief change in the wind after the clouds moved east",
    "fresh chalk marks near the public notice board",
    "a loose hinge that remained safe to use",
    "a quiet interval between the morning and afternoon rounds",
    "several damp leaves gathered beside the drain",
    "a clear view of the path after the temporary barrier was moved",
    "a small difference between the printed map and the new sign",
    "a group of visitors following the marked route",
    "an empty crate returned to the correct storage bay",
    "a steady reading after the instrument was allowed to settle",
    "a repair request already listed in the weekly maintenance book",
    "a short delay while the key holder finished another inspection",
    "a row of seedlings protected from the afternoon sun",
    "a measured queue that cleared before the next scheduled check",
)
FOLLOWUPS = (
    "The note was copied into the routine log before the shift changed.",
    "The team left the route open and added the observation to its next review.",
    "No equipment was moved; the next check remained on the normal schedule.",
    "The clerk marked the entry for comparison with the following day's report.",
    "A second observer confirmed the location without changing the recorded reading.",
    "The crew kept the original notation so later visitors could compare conditions.",
    "The summary was filed with the other ordinary site notes.",
    "The coordinator asked the next shift to use the same reference point.",
)


def load_tokenizer():
    sys.path.insert(0, str(ROOT / "tools"))
    from strata_tokenizer import Tokenizer

    vocab = json.loads((TOKENIZER / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    if any(token is None for token in tokens):
        raise ValueError("tokenizer vocabulary has missing IDs")
    merges = (TOKENIZER / "merges.txt").read_text(encoding="utf-8").splitlines()
    token_types = json.loads((TOKENIZER / "token_type.json").read_text(encoding="utf-8"))
    config = json.loads((TOKENIZER / "tokenizer.json").read_text(encoding="utf-8"))
    return Tokenizer(tokens, merges, token_types, config["pre"], config["special_ids"])


def paragraph(rng: random.Random, number: int) -> str:
    place = rng.choice(PLACES)
    role = rng.choice(ROLES)
    tool = rng.choice(TOOLS)
    observation = rng.choice(OBSERVATIONS)
    followup = rng.choice(FOLLOWUPS)
    weekday = rng.choice(("Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"))
    hour = rng.randrange(6, 22)
    minute = rng.choice((0, 10, 15, 20, 30, 40, 45, 50))
    measurement = rng.randrange(12, 89)
    return (
        f"Field entry {number:05d}: On {weekday} at {hour:02d}:{minute:02d}, "
        f"{role} visited {place} and checked {tool} against the routine sheet. "
        f"The observer noted {observation}; the comparison read {measurement} "
        "units, within the ordinary range recorded for that location. "
        f"{followup}"
    )


def record(label: str, place: str, code: str, description: str) -> str:
    return (
        f"Archive record {label}: {description} At {place}, the sealed register "
        f"lists verification code {code}. Preserve the code exactly as written."
    )


def render(body: str) -> str:
    return CHAT_PREFIX + body + CHAT_SUFFIX


def make_prompt(tk, name: str, target: int, seed: int) -> tuple[str, list[int], dict]:
    rng = random.Random(seed)
    first = record(
        "ORIGIN", "the Alder Quay archive desk", EXPECTED["ORIGIN"],
        "This is the opening record in the dispatch packet.",
    )
    center = record(
        "CENTER", "the Cedar Quarry sample room", EXPECTED["CENTER"],
        "This is the cross-check record filed during the middle review.",
    )
    final = record(
        "FINAL", "the Orbit Kettle return cabinet", EXPECTED["FINAL"],
        "This is the closing record, filed before the packet was sealed.",
    )
    question = (
        "Retrieval task: find the verification code in the ORIGIN, CENTER, and "
        "FINAL archive records. Return exactly three lines, in this order, with "
        "the labels and codes only: ORIGIN: <code>, CENTER: <code>, FINAL: "
        "<code>. Copy each code exactly; do not infer or normalize it."
    )
    parts = [
        (
            f"Survey packet {name}: The following is a chronological collection "
            "of routine field notes. Individual archive records are embedded "
            "among unrelated operational observations."
        ),
        (
            "The notes cover maintenance, visitor flow, weather, and record "
            "handling at several public sites. Values in ordinary entries are "
            "not archive verification codes."
        ),
        first,
    ]
    part_counts = [len(tk.encode(part)) for part in parts]
    separator_count = len(tk.encode("\n\n"))
    wrapper_count = len(tk.encode(render(""), parse_special=True))
    total = wrapper_count + sum(part_counts) + separator_count * (len(parts) - 1)
    middle_at = target // 2
    number = 0

    def add_filler() -> None:
        nonlocal total, number
        text = paragraph(rng, number)
        number += 1
        total += separator_count + len(tk.encode(text))
        parts.append(text)

    while total < middle_at:
        add_filler()
    parts.append(center)
    total += separator_count + len(tk.encode(center))

    tail_count = (
        separator_count + len(tk.encode(final))
        + separator_count + len(tk.encode(question))
    )
    while total + tail_count < target:
        add_filler()
    parts.extend((final, question))

    body = "\n\n".join(parts)
    text = render(body)
    ids = tk.encode(text, parse_special=True)
    while len(ids) < target:
        parts.insert(-2, paragraph(rng, number))
        number += 1
        body = "\n\n".join(parts)
        text = render(body)
        ids = tk.encode(text, parse_special=True)
    if len(ids) + GENERATION_CAP > CONTEXT:
        raise ValueError(
            f"{name}: {len(ids)} input tokens + {GENERATION_CAP} output tokens "
            f"exceed the {CONTEXT}-token context"
        )

    offsets = {}
    for label, marker in (("ORIGIN", first), ("CENTER", center), ("FINAL", final)):
        marker_start = body.index(marker)
        offsets[label] = len(tk.encode(
            CHAT_PREFIX + body[:marker_start], parse_special=True
        ))
    total_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    ids_bytes = (",".join(map(str, ids)) + "\n").encode("ascii")
    ids_hash = hashlib.sha256(ids_bytes).hexdigest()
    metadata = {
        "name": name,
        "seed": seed,
        "target_input_tokens": target,
        "actual_input_tokens": len(ids),
        "generation_cap": GENERATION_CAP,
        "input_plus_generation_cap": len(ids) + GENERATION_CAP,
        "context_tokens": CONTEXT,
        "prompt_text_sha256": total_text_hash,
        "prompt_ids_sha256": ids_hash,
        "needle_offsets_token": offsets,
        "expected": EXPECTED,
        "filler_paragraphs": number,
    }
    return body, ids, metadata


def main() -> int:
    output = OUT / "prompts"
    output.mkdir(parents=True, exist_ok=True)
    tk = load_tokenizer()
    tokenizer_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(TOKENIZER.iterdir()) if path.is_file()
    }
    all_metadata = {
        "tokenizer_pack": str(TOKENIZER),
        "tokenizer_sha256": tokenizer_hashes,
        "chat_template": "user wrapper plus assistant <think> closed, identical to prior 256-token comparison",
        "parse_special": True,
        "stages": [],
    }
    for name, target, seed in STAGES:
        body, ids, metadata = make_prompt(tk, name, target, seed)
        (output / f"{name}.txt").write_text(body + "\n", encoding="utf-8")
        (output / f"{name}.ids").write_text(",".join(map(str, ids)) + "\n", encoding="ascii")
        (output / f"{name}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        actual = len(tk.encode(render(body), parse_special=True))
        if actual != metadata["actual_input_tokens"]:
            raise AssertionError(f"{name}: token count changed during write")
        if not all(code in body for code in EXPECTED.values()):
            raise AssertionError(f"{name}: retrieval needles missing from prompt")
        print(
            f"{name}: {actual} prompt tokens + {GENERATION_CAP} output capacity; "
            f"needle offsets {metadata['needle_offsets_token']}",
            flush=True,
        )
        all_metadata["stages"].append(metadata)
    (output / "manifest.json").write_text(
        json.dumps(all_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
