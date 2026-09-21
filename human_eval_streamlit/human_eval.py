import os
import json
import uuid
import random
import fcntl
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/data/sriparna/swagata/iclr/human")

DATASET_DIR = BASE_DIR / "dataset"

POEM_CSV = DATASET_DIR / "poem.csv"
STORY_CSV = DATASET_DIR / "story.csv"

IMAGE_DIRS = {
    "SD3": BASE_DIR / "sd3_images",
    "FLUX": BASE_DIR / "flux_images",
    "PixArt": BASE_DIR / "pixart_images",
}

OUTPUT_DIR = BASE_DIR / "human_evaluation"

RESULTS_CSV = OUTPUT_DIR / "human_evaluation_results.csv"
AGGREGATED_CSV = OUTPUT_DIR / "human_evaluation_aggregated.csv"
MISSING_IMAGES_CSV = OUTPUT_DIR / "missing_images.csv"
LOCK_PATH = OUTPUT_DIR / "human_evaluation_results.lock"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

MODELS = ["SD3", "FLUX", "PixArt"]

EVALUATION_AXES = [
    "Semantic Fidelity",
    "Attribute Presence",
    "Compositional Correctness",
    "Narrative Fidelity",
]

SLIDER_MIN = 0.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.05


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Human Evaluation",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# IMPORTANT: THIS MUST RUN BEFORE ANY SESSION-STATE ACCESS
# ============================================================

def initialize_session_state():
    """
    Initialize every session-state variable before it is accessed.

    This prevents:
        KeyError: st.session_state has no key "annotator_id"
    """

    defaults = {
        "annotator_id": "",
        "evaluation_order": [],
        "current_position": 0,
        "responses": {},
        "dataset": None,
        "initialized": False,
        "submitted": False,
        "finished": False,
        # tracks which annotator_id the current evaluation_order was
        # built for, so a new annotator (or the same one in a new
        # session) gets a freshly filtered order instead of reusing
        # whatever the first visitor of this session happened to see.
        "order_annotator": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert a value to float safely."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def now_string():
    """Return current timestamp as a string."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_content_type(value):
    """Normalize content type."""

    value = str(value).strip().lower()

    if value == "poem":
        return "poem"

    if value == "story":
        return "story"

    return value


def normalize_annotator_id(value):
    """
    Normalize an annotator id so the same person typing "Akshat",
    "akshat", or " akshat " always maps to one identity instead of
    silently fragmenting into separate "annotators".
    """

    return str(value).strip().lower()


# ============================================================
# DATASET LOADING
# ============================================================

def find_prompt_column(df):
    """
    Automatically identify the text/prompt column.

    Supported candidates:
        prompt, text, poem, story, content, caption, description
    """

    candidates = [
        "prompt",
        "text",
        "poem",
        "story",
        "content",
        "caption",
        "description",
    ]

    lower_to_original = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        "Could not identify the prompt/text column.\n\n"
        f"Available columns: {list(df.columns)}\n\n"
        "Expected one of: "
        + ", ".join(candidates)
    )


def load_single_dataset(csv_path, content_type):
    """
    Load and standardize one dataset.

    IMPORTANT: if the CSV has its own "id" column, that id is used
    directly as prompt_id. Previously this function always recomputed
    prompt_id from row position ("{content_type}_{i:04d}"), which
    silently assumed row i's index always equals the number baked into
    the image filenames. It doesn't: the image set was generated from
    a separate 600-row canonical file whose story ids continue past
    the poem ids (story_0200 .. story_0599, not story_0000 ..
    story_0399), and this file can also be a totally different,
    larger candidate pool with different row content at the same
    position. Recomputing the id caused every prompt shown in the app
    to be paired with the WRONG image (e.g. poem_0008 displayed row 8
    of a 2000-poem pool, "I cradled my newborn daughter...", while
    sd3_images/poem_0008.png is actually "The Lesson" by Maya Angelou
    from the 600-item canonical set). Using the file's own id column
    when present avoids re-deriving something that already exists and
    must match the images exactly.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty:\n{csv_path}"
        )

    prompt_column = find_prompt_column(df)

    standardized = pd.DataFrame()

    standardized["prompt"] = (
        df[prompt_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "id" in df.columns:
        standardized["prompt_id"] = df["id"].astype(str).str.strip()
    else:
        standardized["prompt_id"] = [
            f"{content_type}_{i:04d}"
            for i in range(len(df))
        ]

    standardized["content_type"] = content_type

    standardized = standardized[
        standardized["prompt"].str.len() > 0
    ].copy()

    standardized.reset_index(drop=True, inplace=True)

    return standardized[
        [
            "prompt_id",
            "content_type",
            "prompt",
        ]
    ]


def load_datasets():
    """Load poems and stories."""

    poem_df = load_single_dataset(
        POEM_CSV,
        "poem",
    )

    story_df = load_single_dataset(
        STORY_CSV,
        "story",
    )

    df = pd.concat(
        [
            poem_df,
            story_df,
        ],
        ignore_index=True,
    )

    return df


# ============================================================
# IMAGE RESOLUTION
# ============================================================

def get_image_path(model, prompt_id):
    """
    Resolve image path.

    Expected naming:

        sd3_images/poem_0000.png
        sd3_images/story_0000.png

        flux_images/poem_0000.png
        flux_images/story_0000.png

        pixart_images/poem_0000.png
        pixart_images/story_0000.png
    """

    model_dir = IMAGE_DIRS[model]

    image_path = model_dir / f"{prompt_id}.png"

    if image_path.exists():
        return image_path

    # Try common alternative extensions.
    for extension in [".jpg", ".jpeg", ".webp"]:
        candidate = model_dir / f"{prompt_id}{extension}"

        if candidate.exists():
            return candidate

    return None


# ============================================================
# BUILD EVALUATION ITEMS
# ============================================================

def build_evaluation_items(dataset):
    """
    Create one evaluation item for every
    prompt-model combination.
    """

    items = []

    missing_images = []

    for _, row in dataset.iterrows():

        prompt_id = row["prompt_id"]
        content_type = row["content_type"]

        for model in MODELS:

            image_path = get_image_path(
                model,
                prompt_id,
            )

            item_id = f"{prompt_id}__{model}"

            if image_path is None:

                missing_images.append(
                    {
                        "item_id": item_id,
                        "prompt_id": prompt_id,
                        "content_type": content_type,
                        "model": model,
                        "expected_image": str(
                            IMAGE_DIRS[model]
                            / f"{prompt_id}.png"
                        ),
                    }
                )

                continue

            items.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "content_type": content_type,
                    "prompt": row["prompt"],
                    "model": model,
                    "image_path": str(image_path),
                }
            )

    return items, missing_images


# ============================================================
# SAVE MISSING IMAGE REPORT
# ============================================================

def save_missing_images(missing_images):

    if not missing_images:
        return

    df = pd.DataFrame(missing_images)

    df.to_csv(
        MISSING_IMAGES_CSV,
        index=False,
    )


# ============================================================
# DATASET / ITEM INITIALIZATION (annotator-independent)
# ============================================================

def initialize_dataset():
    """
    Load the dataset and build the full item list ONCE per session.
    Deliberately does not know about the annotator yet -- per-annotator
    filtering and shuffling happens separately in
    ensure_order_for_annotator(), once the annotator id is known.
    """

    if st.session_state["initialized"]:
        return

    dataset = load_datasets()

    items, missing_images = build_evaluation_items(
        dataset
    )

    save_missing_images(
        missing_images
    )

    if not items:
        raise RuntimeError(
            "No evaluation images were found."
        )

    st.session_state["dataset"] = dataset

    st.session_state["all_item_ids"] = [
        item["item_id"] for item in items
    ]

    # Store item metadata separately.
    st.session_state["items"] = {
        item["item_id"]: item
        for item in items
    }

    st.session_state["initialized"] = True


# ============================================================
# PER-ANNOTATOR ORDER: skip items this annotator already did
# ============================================================

def get_completed_item_ids(annotator_id_normalized):
    """
    Read the results CSV (if any) and return the set of item_ids this
    annotator has already submitted, so a new session (reload, restart,
    different device) does not hand them the same items again.
    """

    if not RESULTS_CSV.exists():
        return set()

    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return set()

    if df.empty or "annotator_id" not in df.columns:
        return set()

    df["annotator_id_norm"] = (
        df["annotator_id"].astype(str).str.strip().str.lower()
    )

    mine = df[df["annotator_id_norm"] == annotator_id_normalized]

    return set(mine["item_id"].astype(str))


def ensure_order_for_annotator(annotator_id_raw):
    """
    Build (or rebuild) the shuffled evaluation order for the CURRENT
    annotator, filtering out items they have already completed in a
    previous session. Only rebuilds when the annotator actually changes
    within this session, so mid-session reruns keep the same order.
    """

    annotator_id_normalized = normalize_annotator_id(annotator_id_raw)

    if not annotator_id_normalized:
        return

    if st.session_state.get("order_annotator") == annotator_id_normalized:
        return

    all_item_ids = st.session_state.get("all_item_ids", [])

    completed = get_completed_item_ids(annotator_id_normalized)

    remaining = [
        item_id for item_id in all_item_ids
        if item_id not in completed
    ]

    random.shuffle(remaining)

    st.session_state["evaluation_order"] = remaining
    st.session_state["current_position"] = 0
    st.session_state["responses"] = {}
    st.session_state["order_annotator"] = annotator_id_normalized
    st.session_state["already_completed_count"] = len(completed)


# ============================================================
# GET CURRENT ITEM
# ============================================================

def get_current_item():

    order = st.session_state["evaluation_order"]

    position = st.session_state["current_position"]

    if position >= len(order):
        return None

    item_id = order[position]

    return st.session_state["items"][item_id]


# ============================================================
# CHECK WHETHER CURRENT ITEM HAS BEEN EVALUATED
# ============================================================

def current_item_has_response():

    item = get_current_item()

    if item is None:
        return False

    item_id = item["item_id"]

    return item_id in st.session_state["responses"]


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation():

    """
    Save all completed evaluations to CSV.

    Fixes applied vs. the original version:

      1. De-duplication key is (annotator_id, item_id) -- a real
         identity for "this rater's judgment on this item" -- instead
         of evaluation_id, which used to be a fresh random uuid on
         EVERY save, making drop_duplicates() a permanent no-op. Every
         call used to re-write the annotator's entire session-so-far,
         so a 50-item session wrote 1+2+...+50 = 1275 rows for 50 real
         judgments. Now re-saving the same item just replaces its one
         row (keep="last").

      2. The whole read-modify-write critical section is now wrapped
         in an flock() file lock, so two annotators submitting close
         together can no longer silently clobber each other's rows
         (there was no locking at all before -- `threading` was
         imported but never used).
    """

    annotator_id = st.session_state.get(
        "annotator_id",
        "",
    )

    annotator_id = normalize_annotator_id(annotator_id)

    if not annotator_id:
        st.error(
            "Please enter an Annotator ID before submitting."
        )
        return False

    responses = st.session_state.get(
        "responses",
        {},
    )

    if not responses:
        st.warning(
            "No evaluations have been submitted yet."
        )
        return False

    rows = []

    for item_id, response in responses.items():

        row = {
            "evaluation_id": str(uuid.uuid4()),
            "annotator_id": annotator_id,
            "timestamp": response.get(
                "timestamp",
                now_string(),
            ),
            "item_id": item_id,
            "prompt_id": response.get(
                "prompt_id",
                "",
            ),
            "content_type": response.get(
                "content_type",
                "",
            ),
            "model": response.get(
                "model",
                "",
            ),
        }

        for axis in EVALUATION_AXES:

            row[axis] = safe_float(
                response.get(axis)
            )

        values = [
            row[axis]
            for axis in EVALUATION_AXES
            if row[axis] is not None
        ]

        if values:
            row["human_overall"] = sum(values) / len(values)
        else:
            row["human_overall"] = None

        rows.append(row)

    new_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Critical section: read-modify-write the shared results CSV
    # under an exclusive file lock so concurrent annotators cannot
    # lose each other's rows.
    # --------------------------------------------------------

    LOCK_PATH.touch(exist_ok=True)

    with open(LOCK_PATH, "w") as lock_file:

        fcntl.flock(lock_file, fcntl.LOCK_EX)

        try:

            if RESULTS_CSV.exists():

                try:
                    old_df = pd.read_csv(
                        RESULTS_CSV
                    )

                    combined_df = pd.concat(
                        [
                            old_df,
                            new_df,
                        ],
                        ignore_index=True,
                    )

                except Exception:

                    combined_df = new_df

            else:

                combined_df = new_df

            # Real de-duplication: one row per (annotator_id, item_id).
            # keep="last" means a resubmitted/corrected evaluation
            # overwrites the earlier value for that same item.
            if {"annotator_id", "item_id"} <= set(combined_df.columns):

                combined_df["annotator_id"] = (
                    combined_df["annotator_id"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                combined_df = combined_df.drop_duplicates(
                    subset=["annotator_id", "item_id"],
                    keep="last",
                )

            # Write atomically.
            temp_path = RESULTS_CSV.with_suffix(
                ".tmp.csv"
            )

            combined_df.to_csv(
                temp_path,
                index=False,
            )

            os.replace(
                temp_path,
                RESULTS_CSV,
            )

            # ----------------------------------------------------
            # Aggregation (also inside the lock, so it always reads
            # the just-written, fully de-duplicated file).
            # ----------------------------------------------------

            create_aggregate_results(
                combined_df
            )

        finally:

            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return True


# ============================================================
# AGGREGATION
# ============================================================

def create_aggregate_results(df):

    if df.empty:
        return

    grouping_columns = [
        "prompt_id",
        "content_type",
        "model",
    ]

    aggregation = {}

    for axis in EVALUATION_AXES:

        aggregation[axis] = [
            "mean",
            "std",
            "count",
        ]

    aggregation["human_overall"] = [
        "mean",
        "std",
        "count",
    ]

    aggregated = (
        df.groupby(
            grouping_columns,
            dropna=False,
        )
        .agg(aggregation)
    )

    # Flatten MultiIndex columns.
    flattened_columns = []

    for column in aggregated.columns:

        metric_name = column[0]
        statistic = column[1]

        if statistic == "mean":
            name = metric_name
        else:
            name = f"{metric_name}_{statistic}"

        flattened_columns.append(name)

    aggregated.columns = flattened_columns

    aggregated = aggregated.reset_index()

    aggregated.to_csv(
        AGGREGATED_CSV,
        index=False,
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .prompt-box {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #444;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    .evaluation-note {
        padding: 0.8rem;
        border-radius: 8px;
        background-color: rgba(100, 100, 100, 0.12);
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE DATA
# ============================================================

try:

    initialize_dataset()

except Exception as e:

    st.error(
        "Could not initialize the human evaluation app."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Evaluation Status"
)


# ------------------------------------------------------------
# Annotator ID
# ------------------------------------------------------------

st.sidebar.text_input(
    "Annotator ID",
    key="annotator_id",
    placeholder="Enter your ID",
    help="Use a unique identifier for your evaluation session.",
)


annotator_id = normalize_annotator_id(
    st.session_state.get("annotator_id", "")
)

if annotator_id:
    ensure_order_for_annotator(annotator_id)


# ------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------

dataset = st.session_state["dataset"]

num_poems = int(
    (dataset["content_type"] == "poem").sum()
)

num_stories = int(
    (dataset["content_type"] == "story").sum()
)

num_prompts = len(dataset)

num_models = len(MODELS)

num_images = len(
    st.session_state["evaluation_order"]
)


st.sidebar.markdown(
    f"**Poems:** {num_poems}"
)

st.sidebar.markdown(
    f"**Stories:** {num_stories}"
)

st.sidebar.markdown(
    f"**Total prompts:** {num_prompts}"
)

st.sidebar.markdown(
    f"**Models:** {num_models}"
)

st.sidebar.markdown(
    f"**Evaluation images:** {num_images}"
)

already_done = st.session_state.get("already_completed_count", 0)

if already_done:
    st.sidebar.markdown(
        f"**Already completed by you (previous session):** {already_done}"
    )


# ------------------------------------------------------------
# Progress
# ------------------------------------------------------------

current_position = st.session_state[
    "current_position"
]

completed_count = len(
    st.session_state["responses"]
)

total_items = len(
    st.session_state["evaluation_order"]
)

st.sidebar.markdown(
    f"**Completed this session:** {completed_count} / {total_items}"
)

if total_items > 0:

    st.sidebar.progress(
        min(
            completed_count / total_items,
            1.0,
        )
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    "Human Evaluation of Text-to-Image Generation"
    "</div>",
    unsafe_allow_html=True,
)


st.markdown(
    """
    Please evaluate the generated image with respect to the
    given creative text. The model identity is intentionally
    hidden during evaluation.
    """
)


# ============================================================
# REQUIRE ANNOTATOR ID
# ============================================================

if not annotator_id:

    st.info(
        "Please enter your Annotator ID in the sidebar to begin."
    )

    st.stop()


# ============================================================
# CHECK COMPLETION
# ============================================================

current_item = get_current_item()


if current_item is None:

    if total_items == 0:
        st.success(
            "You have already evaluated every available image. "
            "Nothing left to rate -- thank you!"
        )
    else:
        st.success(
            "You have completed all available evaluations."
        )

        st.balloons()

    st.markdown(
        "Thank you for participating in the evaluation."
    )

    # Save one final time.
    try:

        save_evaluation()

    except Exception as e:

        st.error(
            "The final results could not be saved."
        )

        st.exception(e)

    st.stop()


# ============================================================
# CURRENT ITEM
# ============================================================

item_id = current_item["item_id"]

prompt_id = current_item["prompt_id"]

content_type = current_item["content_type"]

prompt = current_item["prompt"]

image_path = current_item["image_path"]


# ============================================================
# PROGRESS INFORMATION
# ============================================================

st.markdown(
    f"""
    **Evaluation {current_position + 1} of {total_items}**
    """
)

st.markdown(
    f"**Content type:** {content_type.capitalize()}"
)


# ============================================================
# PROMPT
# ============================================================

st.markdown(
    "### Creative Text"
)

st.markdown(
    f"""
    <div class="prompt-box">
    {prompt}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE
# ============================================================

st.markdown(
    "### Generated Image"
)

if not os.path.exists(image_path):

    st.error(
        f"Image not found: {image_path}"
    )

    st.stop()


st.image(
    image_path,
    use_container_width=True,
)


# ============================================================
# EVALUATION INSTRUCTIONS
# ============================================================

st.markdown(
    """
    <div class="evaluation-note">

    <b>Scoring:</b> Use a value from 0 to 1 for each criterion.

    <br><br>

    <b>0</b> = completely unsatisfactory

    <br>

    <b>1</b> = completely satisfactory

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EVALUATION FORM
# ============================================================

# Use a form so sliders do not trigger submission/rerun
# until the evaluator explicitly presses Submit.

with st.form(
    key=f"evaluation_form_{item_id}"
):

    st.markdown(
        "### Evaluation Criteria"
    )

    scores = {}

    scores["Semantic Fidelity"] = st.slider(
        "Semantic Fidelity",
        min_value=SLIDER_MIN,
        max_value=SLIDER_MAX,
        value=0.50,
        step=SLIDER_STEP,
        help=(
            "How faithfully does the image represent "
            "the meaning and content of the text?"
        ),
    )

    scores["Attribute Presence"] = st.slider(
        "Attribute Presence",
        min_value=SLIDER_MIN,
        max_value=SLIDER_MAX,
        value=0.50,
        step=SLIDER_STEP,
        help=(
            "Are the important entities, attributes, "
            "objects, or concepts specified in the text "
            "present in the image?"
        ),
    )

    scores["Compositional Correctness"] = st.slider(
        "Compositional Correctness",
        min_value=SLIDER_MIN,
        max_value=SLIDER_MAX,
        value=0.50,
        step=SLIDER_STEP,
        help=(
            "Are relationships between objects, entities, "
            "and attributes represented correctly?"
        ),
    )

    scores["Narrative Fidelity"] = st.slider(
        "Narrative Fidelity",
        min_value=SLIDER_MIN,
        max_value=SLIDER_MAX,
        value=0.50,
        step=SLIDER_STEP,
        help=(
            "For stories/poems, how well does the image "
            "capture the overall narrative or scene?"
        ),
    )

    reviewed_confirmation = st.checkbox(
        "I looked at the image and set these scores intentionally "
        "(not left at the default).",
    )

    submitted = st.form_submit_button(
        "Submit Evaluation",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# HANDLE SUBMISSION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Require the explicit "I actually rated this" confirmation,
    # since every slider defaults to 0.50 and Streamlit forms give
    # no way to tell "left at default" apart from "genuinely neutral".
    # --------------------------------------------------------

    if not reviewed_confirmation:

        st.error(
            "Please check the confirmation box to submit -- this "
            "prevents accidental submissions at the default (0.50) "
            "score on every axis."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate scores
    # --------------------------------------------------------

    invalid_scores = []

    for axis in EVALUATION_AXES:

        value = scores.get(axis)

        if value is None:

            invalid_scores.append(axis)

        elif value < SLIDER_MIN or value > SLIDER_MAX:

            invalid_scores.append(axis)

    if invalid_scores:

        st.error(
            "Invalid score(s): "
            + ", ".join(invalid_scores)
        )

        st.stop()


    # --------------------------------------------------------
    # Compute overall score
    # --------------------------------------------------------

    overall = sum(
        scores[axis]
        for axis in EVALUATION_AXES
    ) / len(EVALUATION_AXES)


    # --------------------------------------------------------
    # Store response in session state
    # --------------------------------------------------------

    response = {
        "prompt_id": prompt_id,
        "content_type": content_type,
        "model": current_item["model"],

        "Semantic Fidelity": scores[
            "Semantic Fidelity"
        ],

        "Attribute Presence": scores[
            "Attribute Presence"
        ],

        "Compositional Correctness": scores[
            "Compositional Correctness"
        ],

        "Narrative Fidelity": scores[
            "Narrative Fidelity"
        ],

        "human_overall": overall,

        "timestamp": now_string(),
    }


    st.session_state["responses"][
        item_id
    ] = response


    # --------------------------------------------------------
    # Move to next item
    # --------------------------------------------------------

    st.session_state["current_position"] += 1


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    try:

        success = save_evaluation()

        if success:

            st.success(
                "Evaluation saved successfully."
            )

        else:

            st.warning(
                "Evaluation was recorded in the current "
                "session but could not be saved to disk."
            )

    except Exception as e:

        st.error(
            "An error occurred while saving the evaluation."
        )

        st.exception(e)


    # --------------------------------------------------------
    # Rerun to display next item
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# CURRENT SESSION SUMMARY
# ============================================================

if st.session_state["responses"]:

    st.markdown("---")

    st.markdown(
        "### Current Session"
    )

    session_rows = []

    for item_id_saved, response in (
        st.session_state["responses"].items()
    ):

        session_rows.append(
            {
                "Prompt": response.get(
                    "prompt_id",
                    "",
                ),
                "Type": response.get(
                    "content_type",
                    "",
                ),
                "Overall": round(
                    response.get(
                        "human_overall",
                        0,
                    ),
                    3,
                ),
            }
        )

    session_df = pd.DataFrame(
        session_rows
    )

    st.dataframe(
        session_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

if RESULTS_CSV.exists():

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "### Results"
    )

    try:

        with open(
            RESULTS_CSV,
            "rb",
        ) as f:

            st.sidebar.download_button(
                label="Download Raw Results",
                data=f,
                file_name=(
                    "human_evaluation_results.csv"
                ),
                mime="text/csv",
            )

    except Exception:
        pass


if AGGREGATED_CSV.exists():

    try:

        with open(
            AGGREGATED_CSV,
            "rb",
        ) as f:

            st.sidebar.download_button(
                label="Download Aggregated Results",
                data=f,
                file_name=(
                    "human_evaluation_aggregated.csv"
                ),
                mime="text/csv",
            )

    except Exception:
        pass
