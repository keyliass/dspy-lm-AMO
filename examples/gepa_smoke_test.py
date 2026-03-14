from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import dspy
import dspy_lm_auth


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "afso_requirements_gepa_v1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "gepa_smoke"

THEME_CHOICES = (
    "acces",
    "capacite",
    "comptage",
    "confort",
    "environnement",
    "flux",
    "fonctionnement",
    "horaires",
    "implantation",
    "livrable",
    "locaux",
    "maintenance",
    "mutualisation",
    "performance",
    "planning",
    "reglementaire",
    "stationnement",
    "surete",
)

REQUIREMENT_TYPE_CHOICES = (
    "absence",
    "adresse",
    "classification",
    "echeance",
    "implantation",
    "independance",
    "interdiction",
    "obligation",
    "performance",
    "plage_horaire",
    "quantite",
    "separation_flux",
)

UNIT_CHOICES = (
    "",
    "% du temps",
    "EUR HT",
    "ans",
    "bureaux",
    "classes",
    "controle annuel minimum",
    "degC",
    "dm3/m2",
    "halls",
    "heures",
    "kWhEF/m2/an",
    "m",
    "m/s",
    "m3/(h.m2)",
)

FIELD_WEIGHTS = {
    "answer": 0.25,
    "value": 0.25,
    "theme": 0.15,
    "requirement_type": 0.15,
    "applies_to": 0.10,
    "unit": 0.10,
}


class RequirementAnswer(dspy.Signature):
    """Answer from retrieved project context with a short structured result."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Shortest source-grounded answer fragment.")
    theme: str = dspy.OutputField(desc=f"Use exactly one of: {', '.join(THEME_CHOICES)}.")
    requirement_type: str = dspy.OutputField(
        desc=f"Use exactly one of: {', '.join(REQUIREMENT_TYPE_CHOICES)}."
    )
    value: str = dspy.OutputField(desc="Core extracted content, usually identical to answer.")
    unit: str = dspy.OutputField(
        desc=f"Only an explicit unit from context. Prefer one of: {', '.join(repr(x) for x in UNIT_CHOICES)}."
    )
    applies_to: str = dspy.OutputField(desc="Narrowest explicit object or scope affected by the requirement.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex auth checks and a GEPA extraction benchmark on the house dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the JSONL dataset.")
    parser.add_argument("--login", action="store_true", help="Run dspy_lm_auth.login('codex') before anything else.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "solid"),
        default="solid",
        help="Execution profile. 'solid' uses a more credible GEPA setup than the initial smoke run.",
    )
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip the direct reflection-model auth ping.",
    )
    parser.add_argument(
        "--skip-gepa",
        action="store_true",
        help="Only check auth and dataset loading, without running GEPA compile.",
    )
    parser.add_argument(
        "--student-model",
        default="ollama_chat/qwen3.5:0.8b",
        help="Student model route passed to dspy.LM.",
    )
    parser.add_argument(
        "--student-api-base",
        default="http://127.0.0.1:11434",
        help="API base for the student model.",
    )
    parser.add_argument(
        "--student-api-key",
        default="ollama",
        help="API key for the student model route.",
    )
    parser.add_argument(
        "--student-model-type",
        default="chat",
        help="DSPy model_type for the student model.",
    )
    parser.add_argument(
        "--student-max-tokens",
        type=int,
        default=300,
        help="Max tokens for the student model.",
    )
    parser.add_argument(
        "--student-temperature",
        type=float,
        default=0.0,
        help="Temperature for the student model.",
    )
    parser.add_argument(
        "--reflection-model",
        default="codex/gpt-5.4",
        help="Reflection model route used by GEPA.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Optional cap on the number of train examples. 0 means use all.",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=0,
        help="Optional cap on the number of val examples. 0 means use all.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help="Number of validation examples to preview before and after optimization.",
    )
    parser.add_argument(
        "--auto",
        choices=("light", "medium", "heavy"),
        default=None,
        help="Optional override for the GEPA search budget. If omitted, the selected profile decides.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reports and optimized instructions will be written.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
            rows.append(row)
    return rows


def split_rows(
    rows: list[dict[str, Any]], max_train: int, max_val: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if max_train > 0:
        train_rows = train_rows[:max_train]
    if max_val > 0:
        val_rows = val_rows[:max_val]
    return train_rows, val_rows, test_rows


def make_examples(rows: list[dict[str, Any]]) -> list[dspy.Example]:
    return [
        dspy.Example(
            question=row["question"],
            context=row["context"],
            answer=row["answer"],
            theme=row["theme"],
            requirement_type=row["requirement_type"],
            value=row["value"],
            unit=row["unit"],
            applies_to=row["applies_to"],
        ).with_inputs("question", "context")
        for row in rows
    ]


def dataset_summary(rows: list[dict[str, Any]]) -> str:
    split_counts = Counter(row["split"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    return f"rows={len(rows)} | splits={dict(split_counts)} | categories={dict(category_counts)}"


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[\(\)\[\]\{\},;:!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def token_f1(gold: Any, pred: Any) -> float:
    gold_tokens = re.findall(r"[a-z0-9]+", normalize_text(gold))
    pred_tokens = re.findall(r"[a-z0-9]+", normalize_text(pred))
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum(min(gold_counts[token], pred_counts[token]) for token in gold_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return (2 * precision * recall) / (precision + recall)


def text_field_score(gold: Any, pred: Any) -> float:
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(pred)
    if not gold_norm and not pred_norm:
        return 1.0
    if not gold_norm or not pred_norm:
        return 0.0
    if gold_norm == pred_norm:
        return 1.0
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return 0.85
    return token_f1(gold_norm, pred_norm)


def scope_field_score(gold: Any, pred: Any) -> float:
    return text_field_score(gold, pred)


def label_field_score(gold: Any, pred: Any, choices: tuple[str, ...]) -> float:
    gold_norm = normalize_label(gold)
    pred_norm = normalize_label(pred)
    if not gold_norm and not pred_norm:
        return 1.0
    if not gold_norm or not pred_norm:
        return 0.0
    valid = {normalize_label(choice) for choice in choices}
    if pred_norm not in valid:
        return 0.0
    return 1.0 if gold_norm == pred_norm else 0.0


def weighted_score(field_scores: dict[str, float]) -> float:
    return sum(field_scores[field] * FIELD_WEIGHTS[field] for field in FIELD_WEIGHTS)


def build_feedback(gold: dict[str, Any], pred: dict[str, Any], field_scores: dict[str, float]) -> str:
    parts: list[str] = []
    if field_scores["answer"] < 0.999:
        parts.append(
            f"`answer` should be the smallest exact fragment from context. Expected {gold['answer']!r}, got {pred['answer']!r}."
        )
    if field_scores["value"] < 0.999:
        parts.append(
            f"`value` should carry the same core extracted content as the answer. Expected {gold['value']!r}, got {pred['value']!r}."
        )
    if field_scores["theme"] < 0.999:
        parts.append(
            f"`theme` must use the controlled French ontology and stay source-grounded. Expected {gold['theme']!r}, got {pred['theme']!r}. Allowed values: {', '.join(THEME_CHOICES)}."
        )
    if field_scores["requirement_type"] < 0.999:
        parts.append(
            f"`requirement_type` must match the requested information type with the controlled French ontology. Expected {gold['requirement_type']!r}, got {pred['requirement_type']!r}. Allowed values: {', '.join(REQUIREMENT_TYPE_CHOICES)}."
        )
    if field_scores["unit"] < 0.999:
        if normalize_text(gold["unit"]):
            parts.append(
                f"`unit` should keep only the explicit standalone unit from context. Expected {gold['unit']!r}, got {pred['unit']!r}."
            )
        else:
            parts.append("`unit` should be left empty when no explicit standalone unit appears in the context.")
    if field_scores["applies_to"] < 0.999:
        parts.append(
            f"`applies_to` must be the narrowest explicit object or scope affected by the requirement. Expected {gold['applies_to']!r}, got {pred['applies_to']!r}."
        )
    if not parts:
        return "All fields are correctly extracted and normalized."
    return " ".join(parts)


def requirement_metric(gold: dspy.Example, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None):
    del trace, pred_name, pred_trace
    field_scores = {
        "answer": text_field_score(gold.answer, getattr(pred, "answer", "")),
        "theme": label_field_score(gold.theme, getattr(pred, "theme", ""), THEME_CHOICES),
        "requirement_type": label_field_score(
            gold.requirement_type, getattr(pred, "requirement_type", ""), REQUIREMENT_TYPE_CHOICES
        ),
        "value": text_field_score(gold.value, getattr(pred, "value", "")),
        "unit": label_field_score(gold.unit, getattr(pred, "unit", ""), UNIT_CHOICES),
        "applies_to": scope_field_score(gold.applies_to, getattr(pred, "applies_to", "")),
    }
    score = weighted_score(field_scores)
    feedback = build_feedback(
        gold={
            "answer": gold.answer,
            "theme": gold.theme,
            "requirement_type": gold.requirement_type,
            "value": gold.value,
            "unit": gold.unit,
            "applies_to": gold.applies_to,
        },
        pred={
            "answer": getattr(pred, "answer", ""),
            "theme": getattr(pred, "theme", ""),
            "requirement_type": getattr(pred, "requirement_type", ""),
            "value": getattr(pred, "value", ""),
            "unit": getattr(pred, "unit", ""),
            "applies_to": getattr(pred, "applies_to", ""),
        },
        field_scores=field_scores,
    )
    return dspy.Prediction(score=score, feedback=feedback)


def ensure_auth(login: bool) -> None:
    storage = dspy_lm_auth.get_default_auth_storage()
    if login:
        print("Starting Codex OAuth login...")
        dspy_lm_auth.login("codex", auth_storage=storage)
        print("Login completed.")
        return

    if storage.has_auth("codex"):
        return

    raise RuntimeError(
        "No Codex credential found. Run the script with --login first, or authenticate manually with "
        "dspy_lm_auth.login('codex')."
    )


def build_student_lm(args: argparse.Namespace) -> dspy.LM:
    kwargs: dict[str, Any] = {
        "api_base": args.student_api_base,
        "api_key": args.student_api_key,
        "model_type": args.student_model_type,
        "temperature": args.student_temperature,
        "max_tokens": args.student_max_tokens,
    }
    if args.student_model.startswith("ollama_chat/"):
        kwargs["think"] = False
    return dspy.LM(args.student_model, **kwargs)


def auth_ping(reflection_model: str) -> None:
    lm = dspy.LM(reflection_model)
    result = lm("Reply with exactly: auth ok")
    text = ""
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            text = str(first.get("text", "")).strip()
        else:
            text = str(first).strip()
    else:
        text = str(result).strip()
    print(f"Reflection auth check response: {text}")


def preview_predictions(program: Any, rows: list[dict[str, Any]], title: str, count: int) -> None:
    if not rows or count <= 0:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows[:count]:
        pred = program(question=row["question"], context=row["context"])
        print(f"[{row['id']}] question: {row['question']}")
        print(f"  gold answer: {row['answer']}")
        print(f"  pred answer: {getattr(pred, 'answer', '')}")
        print(f"  pred theme: {getattr(pred, 'theme', '')}")
        print(f"  pred type: {getattr(pred, 'requirement_type', '')}")


def evaluate_program(program: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_example: list[dict[str, Any]] = []
    if not rows:
        return {
            "example_count": 0,
            "average_score": 0.0,
            "field_accuracy": {},
            "examples": per_example,
        }

    field_totals = Counter()
    total_score = 0.0

    for row in rows:
        pred = program(question=row["question"], context=row["context"])
        field_scores = {
            "answer": text_field_score(row["answer"], getattr(pred, "answer", "")),
            "theme": label_field_score(row["theme"], getattr(pred, "theme", ""), THEME_CHOICES),
            "requirement_type": label_field_score(
                row["requirement_type"], getattr(pred, "requirement_type", ""), REQUIREMENT_TYPE_CHOICES
            ),
            "value": text_field_score(row["value"], getattr(pred, "value", "")),
            "unit": label_field_score(row["unit"], getattr(pred, "unit", ""), UNIT_CHOICES),
            "applies_to": scope_field_score(row["applies_to"], getattr(pred, "applies_to", "")),
        }
        score = weighted_score(field_scores)
        total_score += score
        for name, similarity in field_scores.items():
            field_totals[name] += similarity

        per_example.append(
            {
                "id": row["id"],
                "question": row["question"],
                "score": score,
                "field_scores": field_scores,
                "gold": {
                    "answer": row["answer"],
                    "theme": row["theme"],
                    "requirement_type": row["requirement_type"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "applies_to": row["applies_to"],
                },
                "pred": {
                    "answer": getattr(pred, "answer", ""),
                    "theme": getattr(pred, "theme", ""),
                    "requirement_type": getattr(pred, "requirement_type", ""),
                    "value": getattr(pred, "value", ""),
                    "unit": getattr(pred, "unit", ""),
                    "applies_to": getattr(pred, "applies_to", ""),
                },
            }
        )

    field_accuracy = {
        field: field_totals[field] / len(rows)
        for field in ("answer", "theme", "requirement_type", "value", "unit", "applies_to")
    }
    return {
        "example_count": len(rows),
        "average_score": total_score / len(rows),
        "field_accuracy": field_accuracy,
        "examples": per_example,
    }


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    train_count: int,
    before_val_eval: dict[str, Any],
    after_val_eval: dict[str, Any],
    before_test_eval: dict[str, Any],
    after_test_eval: dict[str, Any],
    gepa_config: dict[str, Any],
    optimized_instructions: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gepa_smoke_report.json"
    instructions_path = output_dir / "optimized_instructions.txt"

    report = {
        "dataset": str(dataset_path),
        "student_model": args.student_model,
        "student_api_base": args.student_api_base,
        "reflection_model": args.reflection_model,
        "profile": args.profile,
        "gepa_config": gepa_config,
        "train_count": train_count,
        "val_count": before_val_eval["example_count"],
        "test_count": before_test_eval["example_count"],
        "before_val": before_val_eval,
        "after_val": after_val_eval,
        "before_test": before_test_eval,
        "after_test": after_test_eval,
        "val_score_delta": after_val_eval["average_score"] - before_val_eval["average_score"],
        "test_score_delta": after_test_eval["average_score"] - before_test_eval["average_score"],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    instructions_path.write_text(optimized_instructions.strip() + "\n", encoding="utf-8")
    return {
        "report": report_path,
        "instructions": instructions_path,
    }


def print_eval_summary(title: str, evaluation: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Average score: {evaluation['average_score']:.3f} on {evaluation['example_count']} examples")
    print("Field similarity:")
    for field, score in evaluation["field_accuracy"].items():
        print(f"  - {field}: {score:.3f}")


def build_gepa_config(args: argparse.Namespace) -> dict[str, Any]:
    profile_defaults: dict[str, dict[str, Any]] = {
        "smoke": {
            "auto": "light",
            "reflection_minibatch_size": 3,
            "add_format_failure_as_feedback": False,
            "track_stats": False,
            "track_best_outputs": False,
            "num_threads": 1,
            "max_merge_invocations": 5,
        },
        "solid": {
            "reflection_minibatch_size": 4,
            "add_format_failure_as_feedback": True,
            "track_stats": True,
            "track_best_outputs": False,
            "num_threads": 1,
            "max_merge_invocations": 6,
            "max_metric_calls": 240,
        },
    }
    config = dict(profile_defaults[args.profile])
    if args.auto is not None:
        config.pop("max_metric_calls", None)
        config.pop("max_full_evals", None)
        config["auto"] = args.auto
    return config


def main() -> int:
    args = parse_args()

    print(f"Dataset: {args.dataset}")
    rows = load_rows(args.dataset)
    print(f"Loaded dataset: {dataset_summary(rows)}")

    train_rows, val_rows, test_rows = split_rows(rows, args.max_train, args.max_val)
    if not train_rows or not val_rows:
        raise RuntimeError("The selected dataset split is empty. Adjust --max-train or --max-val.")
    print(f"Using train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    ensure_auth(login=args.login)

    dspy_lm_auth.install()

    if not args.skip_auth_check:
        auth_ping(args.reflection_model)

    if args.skip_gepa:
        print("Skipping GEPA compile as requested.")
        return 0

    student_lm = build_student_lm(args)
    reflection_lm = dspy.LM(args.reflection_model)
    dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())

    trainset = make_examples(train_rows)
    valset = make_examples(val_rows)
    program = dspy.Predict(RequirementAnswer)

    preview_predictions(program, val_rows, "Before optimization", args.preview_count)
    before_val_eval = evaluate_program(program, val_rows)
    before_test_eval = evaluate_program(program, test_rows)
    print_eval_summary("Validation before optimization", before_val_eval)
    print_eval_summary("Test before optimization", before_test_eval)

    gepa_config = build_gepa_config(args)
    print(f"\nGEPA config: {gepa_config}")
    gepa = dspy.GEPA(
        metric=requirement_metric,
        reflection_lm=reflection_lm,
        **gepa_config,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    preview_predictions(optimized, val_rows, "After optimization", args.preview_count)
    after_val_eval = evaluate_program(optimized, val_rows)
    after_test_eval = evaluate_program(optimized, test_rows)
    print_eval_summary("Validation after optimization", after_val_eval)
    print_eval_summary("Test after optimization", after_test_eval)

    instructions = getattr(optimized, "signature", None)
    optimized_instructions = ""
    if instructions is not None and hasattr(instructions, "instructions"):
        optimized_instructions = str(instructions.instructions)
        print("\nOptimized instructions")
        print("----------------------")
        print(optimized_instructions)

    paths = write_outputs(
        output_dir=args.output_dir,
        args=args,
        dataset_path=args.dataset,
        train_count=len(train_rows),
        before_val_eval=before_val_eval,
        after_val_eval=after_val_eval,
        before_test_eval=before_test_eval,
        after_test_eval=after_test_eval,
        gepa_config=gepa_config,
        optimized_instructions=optimized_instructions,
    )

    print("\nArtifacts")
    print("---------")
    print(f"Report: {paths['report']}")
    print(f"Instructions: {paths['instructions']}")
    print(f"Validation score delta: {after_val_eval['average_score'] - before_val_eval['average_score']:.3f}")
    print(f"Test score delta: {after_test_eval['average_score'] - before_test_eval['average_score']:.3f}")

    print("\nGEPA smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
