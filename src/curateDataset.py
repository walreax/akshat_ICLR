"""
Local Gemma metaphor-dataset curation.

Run:
    python curateDataset_gemma.py

No OpenAI API calls.

Model:
    google/gemma-1.1-7b-it

Pipeline:
    PoemSum/VIST
        ↓
    10-work batches
        ↓
    local Gemma inference
        ↓
    JSON metaphor annotations
        ↓
    immediate cache
        ↓
    rank ONLY by metaphor_count
        ↓
    top 500 PoemSum + top 500 VIST

Notes:
- Gemma 1.1 7B IT is used rather than the old base gemma-7b because
  this is an instruction-following annotation task.
- The actual GPU inference batch is smaller than 10 by default because
  7B-class models can OOM if 10 long works are tokenized simultaneously.
- BATCH_SIZE=10 means "10 works per logical batch"; LOCAL_INFER_BATCH
  controls how many are fed to the GPU at once.
- Each successful work is written to the JSONL cache immediately.
- Existing cache is reused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-1.1-7b-it"

# Public Hugging Face access still requires accepting Google's Gemma terms.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Logical curation batch.
BATCH_SIZE = 10

# Actual simultaneous GPU inference.
# Increase to 4 if your GPU has enough VRAM.
LOCAL_INFER_BATCH = 2

# Keep prompt lengths bounded for Gemma's context.
MAX_INPUT_TOKENS = 7000
MAX_NEW_TOKENS = 900

TEMPERATURE = 0.0

# First test.
TRAIN_LIMIT = None
# Example:
# TRAIN_LIMIT = 100

TARGET_PER_SOURCE = 500

OUTPUT_FILE = "metaphor_dataset_1000.json"
CACHE_FILE = "metaphor_work_cache_gemma.jsonl"
COUNTS_FILE = "metaphor_counts_gemma.json"
RANKING_FILE = "metaphor_ranking_gemma.json"

MODEL_DTYPE = torch.bfloat16

POEMSUM_BASE = (
    "https://raw.githubusercontent.com/"
    "Ridwan230/PoemSum/main/Dataset/"
)

POEMSUM_FILES = [
    "poemsum_train.csv",
    "poemsum_valid.csv",
    "poemsum_test.csv",
]

VIST_URL = (
    "https://visionandlanguage.net/"
    "VIST/json_files/story-in-sequence/"
    "SIS-with-labels.tar.gz"
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)


# ============================================================
# DATA TYPES / HELPERS
# ============================================================

@dataclass
class Work:
    work_id: str
    name_of_work: str
    content: str
    author: str | None
    source: str
    original_index: int


def clean(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x).strip()


def norm(x: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(x).lower(),
    ).strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:12]


def find_value(
    row: dict[str, Any],
    candidates,
):
    if not isinstance(row, dict):
        return None

    normalized = {
        norm(k): k
        for k in row
    }

    for candidate in candidates:
        c = norm(candidate)

        if c in normalized:
            return row[
                normalized[c]
            ]

    for nk, original in normalized.items():
        for candidate in candidates:
            c = norm(candidate)

            if c in nk or nk in c:
                return row[original]

    return None


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download(
    url: str,
    path: Path,
):
    print(
        "Downloading:",
        url,
    )

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": (
            "https://visionandlanguage.net/"
            "VIST/dataset.html"
        ),
        "Accept": "*/*",
    }

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=120,
            allow_redirects=True,
        ) as response:

            if response.status_code < 400:
                with path.open(
                    "wb"
                ) as f:
                    for chunk in response.iter_content(
                        1024 * 1024
                    ):
                        if chunk:
                            f.write(chunk)

                return

            print(
                "requests returned",
                response.status_code,
                "- trying curl",
            )

    except Exception as exc:
        print(
            "requests failed:",
            exc,
            "- trying curl",
        )

    import shutil
    import subprocess

    curl = shutil.which(
        "curl"
    )

    if not curl:
        raise RuntimeError(
            "curl is required for fallback downloading."
        )

    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "4",
            "-A",
            USER_AGENT,
            "-e",
            "https://visionandlanguage.net/VIST/dataset.html",
            "-o",
            str(path),
            url,
        ],
        check=True,
    )


# ============================================================
# POEMSUM
# ============================================================

def load_poemsum(
    temp_dir: Path,
):
    works = []
    seen = set()

    for filename in POEMSUM_FILES:
        path = (
            temp_dir / filename
        )

        download(
            POEMSUM_BASE + filename,
            path,
        )

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        for i, row in enumerate(
            rows
        ):
            title = clean(
                row.get("Title")
                or row.get("title")
            )

            author = clean(
                row.get("Poet")
                or row.get("poet")
                or row.get("Author")
            )

            content = clean(
                row.get("ctext")
                or row.get("text")
                or row.get("poem")
            )

            if not content:
                continue

            key = (
                title.lower(),
                content,
            )

            if key in seen:
                continue

            seen.add(key)

            works.append(
                Work(
                    work_id=(
                        f"poem_sum:"
                        f"{filename}:"
                        f"{i}:"
                        f"{short_hash(title + '|' + content)}"
                    ),
                    name_of_work=(
                        title
                        or f"Poem {i + 1}"
                    ),
                    content=content,
                    author=author or None,
                    source="poem_sum",
                    original_index=len(
                        works
                    ),
                )
            )

    return works


# ============================================================
# VIST SIS
# ============================================================

def read_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def extract_sis_records(
    obj,
):
    if isinstance(
        obj,
        dict,
    ):
        annotations = obj.get(
            "annotations"
        )

        if isinstance(
            annotations,
            list,
        ):
            records = []

            for group in annotations:
                if isinstance(
                    group,
                    list,
                ):
                    items = [
                        x
                        for x in group
                        if isinstance(
                            x,
                            dict,
                        )
                    ]

                    if items:
                        records.append(
                            items
                        )

                elif isinstance(
                    group,
                    dict,
                ):
                    records.append(
                        group
                    )

            return records

        for key in (
            "data",
            "stories",
            "story",
            "records",
        ):
            value = obj.get(
                key
            )

            if isinstance(
                value,
                list,
            ):
                return [
                    x
                    for x in value
                    if isinstance(
                        x,
                        dict,
                    )
                ]

    if isinstance(
        obj,
        list,
    ):
        records = []

        for group in obj:
            if isinstance(
                group,
                list,
            ):
                items = [
                    x
                    for x in group
                    if isinstance(
                        x,
                        dict,
                    )
                ]

                if items:
                    records.append(
                        items
                    )

            elif isinstance(
                group,
                dict,
            ):
                records.append(
                    group
                )

        return records

    return []


def load_vist(
    temp_dir: Path,
):
    archive = (
        temp_dir
        / "SIS-with-labels.tar.gz"
    )

    download(
        VIST_URL,
        archive,
    )

    extracted = (
        temp_dir
        / "vist"
    )

    extracted.mkdir()

    print(
        "Extracting VIST SIS..."
    )

    with tarfile.open(
        archive,
        "r:gz",
    ) as tar:
        tar.extractall(
            extracted,
            filter="data",
        )

    works = []
    seen = set()

    for path in extracted.rglob(
        "*.json"
    ):
        try:
            obj = read_json(
                path
            )
        except Exception:
            continue

        for record in extract_sis_records(
            obj
        ):
            if not isinstance(
                record,
                list,
            ):
                continue

            items = list(
                record
            )

            def order_key(
                item
            ):
                value = find_value(
                    item,
                    [
                        "image_order",
                        "sentence_order",
                        "order",
                    ],
                )

                try:
                    return int(
                        value
                    )
                except Exception:
                    return 999

            items.sort(
                key=order_key
            )

            sentences = []

            for item in items:
                sentence = clean(
                    find_value(
                        item,
                        [
                            "storytext",
                            "story_text",
                            "sentence",
                            "caption",
                            "text",
                        ],
                    )
                )

                if sentence:
                    sentences.append(
                        sentence
                    )

            if not sentences:
                continue

            story_id = clean(
                find_value(
                    items[0],
                    [
                        "story_id",
                        "sequence_id",
                        "album_id",
                        "storylet_id",
                    ],
                )
            )

            content = " ".join(
                sentences[:5]
            ).strip()

            key = (
                story_id,
                content,
            )

            if key in seen:
                continue

            seen.add(key)

            works.append(
                Work(
                    work_id=(
                        f"visual_story:"
                        f"{story_id or len(works)}:"
                        f"{short_hash(content)}"
                    ),
                    name_of_work=(
                        f"Visual Story "
                        f"{story_id or len(works) + 1}"
                    ),
                    content=content,
                    author=None,
                    source="visual_story",
                    original_index=len(
                        works
                    ),
                )
            )

    return works


# ============================================================
# GEMMA MODEL
# ============================================================

def load_model():
    print(
        "\nLoading Gemma:"
        f" {MODEL_ID}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",
    )

    model.eval()

    print(
        "Gemma loaded."
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    work: Work,
):
    # IMPORTANT: TEXT is deliberately LAST, after the JSON schema and
    # rules, not before them. MAX_INPUT_TOKENS truncation (see
    # analyze_batch) cuts from the tail of the tokenized prompt. With
    # the schema/rules after a long TEXT, truncation could cut into
    # the instructions themselves instead of the work text -- Gemma
    # would then be left completing a truncated sentence (e.g. a
    # chopped '"source_domain"') instead of answering, producing
    # garbled preamble that can also break naive JSON extraction.
    # Putting TEXT last means truncation, if it happens, only ever
    # clips the work being annotated -- the instructions and schema
    # always survive intact.
    return f"""
You are annotating ONE creative work for a research dataset focused on metaphor.

Return ONLY valid JSON. No markdown and no explanation.

WORK ID:
{work.work_id}

TITLE:
{work.name_of_work}

AUTHOR:
{work.author or "Unknown"}

SOURCE:
{work.source}

Return exactly this JSON structure:

{{
  "id": "{work.work_id}",
  "metaphor_count": 0,
  "real_concepts": [],
  "metaphors": []
}}

Each item in "metaphors" must have:
- "text"
- "source_domain"
- "target_domain"
- "real_concept"
- "interpretation"
- "confidence"

Rules:
- Count DISTINCT GENUINE METAPHORICAL EXPRESSIONS.
- Be conservative.
- Do not count ordinary literal description.
- Do not automatically count similes.
- Do not inflate the count with conventional idioms unless clearly metaphorical in context.
- Personification may count when clearly metaphorical.
- Count distinct expressions, not repeated occurrences.
- "metaphor_count" MUST equal the number of objects in "metaphors".
- "real_concepts" should contain the underlying concepts represented by the metaphors.

TEXT:
{work.content}
""".strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Gemma sometimes echoes the prompt/instructions and then emits the
    requested JSON. Scan for balanced JSON objects and select the object
    matching the metaphor-annotation schema.
    """
    text = text.strip()

    # Remove markdown fences anywhere in the response.
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("```", "")

    candidates = []
    depth = 0
    start_idx = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            # Only start honoring string state once we're already
            # inside a candidate object (depth > 0). Stray/unbalanced
            # quote characters in preamble text Gemma sometimes
            # echoes before its real answer (e.g. a truncated
            # '"source_domain"' fragment) must never be allowed to
            # desync string-tracking for the rest of the text -- that
            # was silently swallowing perfectly valid JSON objects by
            # leaving the scanner permanently convinced it was still
            # "inside a string" once it reached them.
            if depth > 0:
                in_string = True
        elif ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidates.append(
                        text[start_idx:i + 1]
                    )
                    start_idx = None

    required = {
        "metaphor_count",
        "real_concepts",
        "metaphors",
    }

    # Prefer the LAST matching object, because Gemma often echoes the
    # requested schema first and then produces its actual answer later.
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if (
            isinstance(obj, dict)
            and required.issubset(obj.keys())
        ):
            return obj

    raise ValueError(
        "No valid metaphor-annotation JSON object found.\n"
        f"Gemma output:\n{text}"
    )

def normalize_result(
    result,
    work: Work,
):
    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "Gemma output was not an object."
        )

    result["id"] = work.work_id

    result.setdefault(
        "real_concepts",
        [],
    )

    result.setdefault(
        "metaphors",
        [],
    )

    result["metaphor_count"] = int(
        result.get(
            "metaphor_count",
            len(
                result["metaphors"]
            ),
        )
    )

    # If Gemma gives inconsistent count, trust the explicit list.
    # The final dataset must remain internally consistent.
    result["metaphor_count"] = len(
        result["metaphors"]
    )

    return result


# ============================================================
# LOCAL BATCH INFERENCE
# ============================================================

def analyze_batch(
    tokenizer,
    model,
    batch: list[Work],
):
    outputs = {}

    for start in range(
        0,
        len(batch),
        LOCAL_INFER_BATCH,
    ):
        micro_batch = batch[
            start:start + LOCAL_INFER_BATCH
        ]

        chats = [
            [
                {
                    "role": "user",
                    "content": build_prompt(
                        work
                    ),
                }
            ]
            for work in micro_batch
        ]

        rendered = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            for chat in chats
        ]

        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(
            model.device
        )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lengths = (
            inputs["attention_mask"]
            .sum(dim=1)
            .tolist()
        )

        for i, work in enumerate(
            micro_batch
        ):
            generated_tokens = (
                generated[i][
                    input_lengths[i]:
                ]
            )

            text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            try:
                result = extract_json(
                    text
                )

                result = normalize_result(
                    result,
                    work,
                )

                outputs[
                    work.work_id
                ] = result

            except Exception as exc:
                raise RuntimeError(
                    f"Gemma failed to produce valid JSON "
                    f"for {work.work_id}.\n"
                    f"Raw output:\n{text}\n"
                    f"Error: {exc}"
                ) from exc

    return outputs


# ============================================================
# CACHE
# ============================================================

def load_cache(
    path: Path,
):
    cache = {}

    if not path.exists():
        return cache

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(
                    line
                )

                if item.get(
                    "id"
                ):
                    cache[
                        item["id"]
                    ] = item

            except Exception:
                continue

    return cache


def append_cache(
    path: Path,
    item,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


# ============================================================
# RANKING
# ============================================================

def rank(
    works,
    counts,
):
    # ONLY metaphor_count.
    # Stable sorting preserves source order among ties.
    ordered = sorted(
        works,
        key=lambda work:
            counts[
                work.work_id
            ],
        reverse=True,
    )

    ranks = {}
    last_count = None
    current_rank = 0

    for position, work in enumerate(
        ordered,
        start=1,
    ):
        count = counts[
            work.work_id
        ]

        if count != last_count:
            last_count = count
            current_rank = position

        ranks[
            work.work_id
        ] = current_rank

    ranking = []

    for position, work in enumerate(
        ordered,
        start=1,
    ):
        ranking.append(
            {
                "id":
                    work.work_id,
                "metaphor_count":
                    counts[
                        work.work_id
                    ],
                "rank":
                    ranks[
                        work.work_id
                    ],
                "selected":
                    position <= TARGET_PER_SOURCE,
            }
        )

    return (
        ordered[
            :TARGET_PER_SOURCE
        ],
        ranking,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for local Gemma inference."
        )

    print()
    print("=" * 72)
    print("LOCAL GEMMA METAPHOR DATASET CURATION")
    print("=" * 72)
    print(
        f"Model: {MODEL_ID}"
    )
    print(
        f"Logical batch: {BATCH_SIZE}"
    )
    print(
        f"GPU micro-batch: {LOCAL_INFER_BATCH}"
    )
    print(
        "Ranking criterion: metaphor_count ONLY"
    )

    if TRAIN_LIMIT is not None:
        print(
            f"LIMITED TEST RUN: {TRAIN_LIMIT}"
        )

    tokenizer, model = load_model()

    cache_path = Path(
        CACHE_FILE
    )

    cache = load_cache(
        cache_path
    )

    print(
        f"Cached works: {len(cache):,}"
    )

    # --------------------------------------------------------
    # Download/load data
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="gemma_metaphor_"
    ) as temp_name:

        temp = Path(
            temp_name
        )

        print(
            "\nLoading PoemSum..."
        )

        poemsum = load_poemsum(
            temp
        )

        print(
            f"PoemSum works: {len(poemsum):,}"
        )

        print(
            "\nLoading VIST..."
        )

        vist = load_vist(
            temp
        )

        print(
            f"VIST stories: {len(vist):,}"
        )

        all_works = (
            poemsum + vist
        )

        if TRAIN_LIMIT is not None:
            all_works = all_works[
                :TRAIN_LIMIT
            ]

        print(
            f"\nWorks to analyze: "
            f"{len(all_works):,}"
        )

        # ----------------------------------------------------
        # Local Gemma annotation
        # ----------------------------------------------------

        total = len(
            all_works
        )

        for batch_start in range(
            0,
            total,
            BATCH_SIZE,
        ):
            batch = all_works[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            missing = [
                work
                for work in batch
                if work.work_id not in cache
            ]

            if not missing:
                continue

            print()
            print(
                f"[{batch_start + 1}-"
                f"{min(batch_start + BATCH_SIZE, total)}"
                f" / {total}]"
            )

            started = time.time()

            results = analyze_batch(
                tokenizer,
                model,
                missing,
            )

            for work in missing:
                result = results[
                    work.work_id
                ]

                append_cache(
                    cache_path,
                    result,
                )

                cache[
                    work.work_id
                ] = result

            print(
                f"  analyzed: {len(missing)}"
            )
            print(
                f"  time: "
                f"{time.time() - started:.1f}s"
            )
            print(
                f"  total cached: "
                f"{len(cache):,}"
            )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        counts = {
            work.work_id:
                int(
                    cache[
                        work.work_id
                    ][
                        "metaphor_count"
                    ]
                )
            for work in all_works
        }

        Path(
            COUNTS_FILE
        ).write_text(
            json.dumps(
                counts,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Rank separately
        # ----------------------------------------------------

        poem_selected, poem_ranking = rank(
            poemsum,
            counts,
        )

        vist_selected, vist_ranking = rank(
            vist,
            counts,
        )

        Path(
            RANKING_FILE
        ).write_text(
            json.dumps(
                {
                    "poem_sum":
                        poem_ranking,
                    "visual_story":
                        vist_ranking,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Final dataset
        # ----------------------------------------------------

        selected = (
            poem_selected
            + vist_selected
        )

        # During a limited smoke test we cannot produce 1000.
        if len(selected) < 2 * TARGET_PER_SOURCE:
            print(
                "\nSmoke test finished."
            )
            print(
                "Not enough works for the 1000-item "
                "final dataset."
            )
            return

        final = []

        for work in selected:
            annotation = cache[
                work.work_id
            ]

            final.append(
                {
                    "name_of_work":
                        work.name_of_work,
                    "content":
                        work.content,
                    "author":
                        work.author,
                    "source":
                        work.source,
                    "real_concepts":
                        annotation[
                            "real_concepts"
                        ],
                    "metaphor_count":
                        annotation[
                            "metaphor_count"
                        ],
                    "metaphors":
                        annotation[
                            "metaphors"
                        ],
                }
            )

        Path(
            OUTPUT_FILE
        ).write_text(
            json.dumps(
                final,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("DONE")
        print("=" * 72)
        print(
            "PoemSum selected:",
            len(poem_selected),
        )
        print(
            "VIST selected:",
            len(vist_selected),
        )
        print(
            "Final dataset:",
            OUTPUT_FILE,
        )

    print(
        "\nTemporary source datasets deleted."
    )


if __name__ == "__main__":
    main()