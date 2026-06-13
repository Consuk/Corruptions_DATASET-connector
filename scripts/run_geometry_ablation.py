from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


ERROR_METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log")
ACCURACY_METRICS = ("a1", "a2", "a3")
ALL_METRICS = ERROR_METRICS + ACCURACY_METRICS
MODELS = (
    "monodepth2",
    "monovit",
    "af_sfmlearner",
    "endosfmlearner",
    "endodac",
    "monoiit",
)
DATASETS = ("hamlyn", "c3vd")
CONDITIONS = ("aware", "ablated")


METRIC_ALIASES = {
    "abs rel": "abs_rel",
    "abs_rel": "abs_rel",
    "absrel": "abs_rel",
    "sq rel": "sq_rel",
    "sq_rel": "sq_rel",
    "sqrel": "sq_rel",
    "rmse": "rmse",
    "rmse log": "rmse_log",
    "rmse_log": "rmse_log",
    "rmselog": "rmse_log",
    "a1": "a1",
    "a2": "a2",
    "a3": "a3",
    "delta1": "a1",
    "delta2": "a2",
    "delta3": "a3",
    "d1": "a1",
    "d2": "a2",
    "d3": "a3",
    "δ1": "a1",
    "δ2": "a2",
    "δ3": "a3",
}


@dataclass(frozen=True)
class ModelEvalConfig:
    repo_root: Path
    script: str
    weights: Path
    default_args: tuple[str, ...]
    output_style: str


@dataclass
class RunPlan:
    dataset: str
    model: str
    condition: str
    repo_root: Path
    command: list[str]
    output_csv: Path
    log_path: Path
    manifest_path: Path
    warnings: list[str]


def p(path: str) -> Path:
    return Path(path)


DEFAULT_CONFIGS: dict[str, dict[str, ModelEvalConfig]] = {
    "hamlyn": {
        "monodepth2": ModelEvalConfig(
            repo_root=p("/workspace/Monodepth2"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/monodepth2_hamlyn_weights_19"),
            default_args=(
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--min_depth",
                "1",
                "--max_depth",
                "50",
            ),
            output_style="output_dir",
        ),
        "monovit": ModelEvalConfig(
            repo_root=p("/workspace/monodepth2_monovit"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/monovit_hamlyn_weights_19"),
            default_args=(
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--min_depth",
                "1",
                "--max_depth",
                "50",
            ),
            output_style="output_dir",
        ),
        "af_sfmlearner": ModelEvalConfig(
            repo_root=p("/workspace/repos/AF-SfMLearner"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/afsfmlearner_hamlyn_weights_19"),
            default_args=(
                "--splits_dir",
                "/workspace/repos/AF-SfMLearner/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--min_depth",
                "1",
                "--max_depth",
                "50",
                "--hamlyn_strict_neighbors",
            ),
            output_style="output_dir",
        ),
        "endodac": ModelEvalConfig(
            repo_root=p("/workspace/ENDO-DAC"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/endodac_hamlyn_weights_last"),
            default_args=(
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--min_depth",
                "1",
                "--max_depth",
                "50",
                "--learn_intrinsics",
                "false",
                "--hamlyn_use_intrinsics_file",
                "true",
                "--hamlyn_intrinsics_filename",
                "intrinsics.txt",
            ),
            output_style="output_dir",
        ),
        "monoiit": ModelEvalConfig(
            repo_root=p("/workspace/endo-manydepth/endo-manydepth-master/manydepth"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/monoiit_manydepth_hamlyn_weights_19"),
            default_args=(
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--min_depth",
                "1",
                "--max_depth",
                "50",
            ),
            output_style="output_dir",
        ),
        "endosfmlearner": ModelEvalConfig(
            repo_root=p("/workspace/repos/Endo-SfM-Learner-new-try"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/hamlyn_weights/endosfmlearner_hamlyn_weights"),
            default_args=(
                "--splits_dir",
                "/workspace/repos/Endo-SfM-Learner-new-try/splits",
                "--split",
                "hamlyn",
                "--dataset",
                "hamlyn",
                "--hamlyn_eval_min_depth",
                "1.0",
                "--hamlyn_eval_max_depth",
                "50.0",
            ),
            output_style="output_csv",
        ),
    },
    "c3vd": {
        "monodepth2": ModelEvalConfig(
            repo_root=p("/workspace/Monodepth2"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/monodepth2_c3vd_weights_19"),
            default_args=(
                "--dataset",
                "c3vd",
                "--split",
                "c3vd",
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--height",
                "256",
                "--width",
                "320",
                "--batch_size",
                "12",
                "--min_depth",
                "0.1",
                "--max_depth",
                "100.0",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
            ),
            output_style="output_dir",
        ),
        "monovit": ModelEvalConfig(
            repo_root=p("/workspace/monodepth2_monovit"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/monovit_c3vd_weights_19"),
            default_args=(
                "--dataset",
                "c3vd",
                "--split",
                "c3vd",
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--height",
                "256",
                "--width",
                "320",
                "--batch_size",
                "12",
                "--min_depth",
                "0.1",
                "--max_depth",
                "100.0",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
            ),
            output_style="output_dir",
        ),
        "af_sfmlearner": ModelEvalConfig(
            repo_root=p("/workspace/repos/AF-SfMLearner"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/afsfmlearner_c3vd_weights_19"),
            default_args=(
                "--dataset",
                "c3vd",
                "--split",
                "c3vd",
                "--splits_dir",
                "/workspace/repos/AF-SfMLearner/splits",
                "--height",
                "256",
                "--width",
                "320",
                "--batch_size",
                "12",
                "--min_depth",
                "0.1",
                "--max_depth",
                "100.0",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
                "--c3vd_use_loss_mask",
                "--c3vd_mask_filename",
                "mask.png",
                "--c3vd_mask_erosion",
                "1",
            ),
            output_style="output_dir",
        ),
        "endodac": ModelEvalConfig(
            repo_root=p("/workspace/ENDO-DAC"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/endodac_c3vd_weights_last"),
            default_args=(
                "--dataset",
                "c3vd",
                "--split",
                "c3vd",
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--height",
                "224",
                "--width",
                "280",
                "--batch_size",
                "8",
                "--min_depth",
                "0.1",
                "--max_depth",
                "100.0",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
                "--learn_intrinsics",
                "false",
                "--c3vd_use_intrinsics_file",
                "true",
                "--num_workers",
                "2",
            ),
            output_style="output_dir",
        ),
        "monoiit": ModelEvalConfig(
            repo_root=p("/workspace/endo-manydepth/endo-manydepth-master/manydepth"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/monoiit_manydepth_c3vd_weights_19"),
            default_args=(
                "--split",
                "c3vd",
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--height",
                "256",
                "--width",
                "320",
                "--png",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
            ),
            output_style="output_csv",
        ),
        "endosfmlearner": ModelEvalConfig(
            repo_root=p("/workspace/repos/Endo-SfM-Learner-new-try"),
            script="eval_endovis_corruptions.py",
            weights=p("/workspace/c3vd_weights/endosfmlearner_c3vd_weights"),
            default_args=(
                "--splits_dir",
                "/workspace/Monodepth2/splits",
                "--split",
                "c3vd",
                "--dataset",
                "c3vd",
                "--img_height",
                "288",
                "--img_width",
                "512",
                "--c3vd_eval_min_depth",
                "0.1",
                "--c3vd_eval_max_depth",
                "100.0",
            ),
            output_style="output_csv",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run geometry-aware vs wrong-geometry ablations for Hamlyn and C3VD, "
            "then compute raw metrics, mCE, mDERS, deltas, and a LaTeX table."
        )
    )
    parser.add_argument("--dataset", choices=("all", "hamlyn", "c3vd"), default="all")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", default="results/geometry_ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Do not run evaluators; aggregate CSVs already present in the intermediate directory.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--no-filter-unsupported-args",
        action="store_true",
        help=(
            "Do not inspect evaluator --help output. By default unsupported "
            "optional arguments are removed per repo for compatibility."
        ),
    )

    parser.add_argument(
        "--hamlyn-aware-corruptions-root",
        default="/workspace/datasets/hamlyn/hamlyn_corruptions_test24",
    )
    parser.add_argument(
        "--hamlyn-ablated-corruptions-root",
        default="/workspace/datasets/hamlyn/hamlyn_corruptions_test24_globalK",
        help=(
            "Root used for the global-K Hamlyn condition. If the same images are used, "
            "pass the repo-specific global-K override via --hamlyn-ablated-extra-args."
        ),
    )
    parser.add_argument(
        "--prepare-hamlyn-global-k-corruptions",
        action="store_true",
        help=(
            "Create --hamlyn-ablated-corruptions-root by linking/copying the aware "
            "Hamlyn corruptions and replacing every intrinsics.txt with one global K_0."
        ),
    )
    parser.add_argument(
        "--hamlyn-global-k-source",
        default=None,
        help=(
            "intrinsics.txt used as K_0 for Hamlyn ablation. If omitted, the first "
            "intrinsics.txt found under --hamlyn-aware-corruptions-root is used."
        ),
    )
    parser.add_argument(
        "--hamlyn-global-k-file-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to mirror non-intrinsics files when preparing Hamlyn global-K corruptions.",
    )
    parser.add_argument(
        "--c3vd-aware-corruptions-root",
        default="/workspace/datasets/c3vd_corrupted",
    )
    parser.add_argument(
        "--c3vd-ablated-corruptions-root",
        default="/workspace/datasets/c3vd_corrupted_raw",
        help="Corruptions generated from raw omnidirectional C3VD frames.",
    )
    parser.add_argument(
        "--c3vd-raw-data-root",
        default="/workspace/datasets/c3vd_consuk",
        help="Original non-preprocessed C3VD root used to generate raw ablated corruptions.",
    )
    parser.add_argument(
        "--c3vd-split-file",
        default="/workspace/Monodepth2/splits/c3vd/test_files.txt",
        help="C3VD test split used when generating raw ablated corruptions.",
    )
    parser.add_argument(
        "--prepare-c3vd-raw-corruptions",
        action="store_true",
        help=(
            "Generate --c3vd-ablated-corruptions-root from --c3vd-raw-data-root "
            "before evaluation, using generate_corruptions_from_testlist.py."
        ),
    )
    parser.add_argument("--hamlyn-aware-extra-args", default="")
    parser.add_argument(
        "--hamlyn-ablated-extra-args",
        default="",
        help="Repo-specific flags that force the global K_0 Hamlyn evaluation.",
    )
    parser.add_argument("--c3vd-aware-extra-args", default="")
    parser.add_argument(
        "--c3vd-ablated-extra-args",
        default="",
        help=(
            "Repo-specific flags for raw-omni-as-pinhole C3VD evaluation, such as "
            "a raw data_path or disabling pinhole preprocessing."
        ),
    )
    parser.add_argument(
        "--allow-empty-ablation-overrides",
        action="store_true",
        help=(
            "Allow ablated Hamlyn/C3VD runs without explicit extra args. Use only when "
            "the ablated root itself fully encodes the wrong-geometry protocol."
        ),
    )
    return parser.parse_args()


def selected_datasets(raw: str) -> list[str]:
    return list(DATASETS) if raw == "all" else [raw]


def selected_corruptions(raw: list[str]) -> Optional[set[str]]:
    if len(raw) == 1 and raw[0].lower() == "all":
        return None
    return {normalize_text(item) for item in raw}


def normalize_text(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def split_extra_args(value: str) -> list[str]:
    return shlex.split(value) if value.strip() else []


def condition_corruptions_root(args: argparse.Namespace, dataset: str, condition: str) -> Path:
    key = f"{dataset}_{condition}_corruptions_root"
    return Path(getattr(args, key)).expanduser()


def condition_extra_args(args: argparse.Namespace, dataset: str, condition: str) -> list[str]:
    key = f"{dataset}_{condition}_extra_args"
    return split_extra_args(getattr(args, key))


def output_csv_for(plan_dir: Path, config: ModelEvalConfig) -> tuple[Path, list[str]]:
    if config.output_style == "output_csv":
        csv_path = plan_dir / "summary_by_severity.csv"
        return csv_path, ["--output_csv", str(csv_path)]

    csv_path = plan_dir / "summary_by_severity.csv"
    return csv_path, [
        "--run_name",
        plan_dir.name,
        "--output_dir",
        str(plan_dir.parent),
        "--summary_filename",
        "summary_by_severity.csv",
        "--per_corruption_filename",
        "summary_by_corruption.csv",
        "--global_avg_filename",
        "summary_global.csv",
    ]


def build_plan(args: argparse.Namespace) -> list[RunPlan]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    plans: list[RunPlan] = []
    for dataset in selected_datasets(args.dataset):
        for condition in CONDITIONS:
            root = condition_corruptions_root(args, dataset, condition)
            extra_args = condition_extra_args(args, dataset, condition)
            if condition == "ablated" and not extra_args and not args.allow_empty_ablation_overrides:
                ablated_root_differs = root != condition_corruptions_root(args, dataset, "aware")
                if dataset == "hamlyn" and not (
                    args.prepare_hamlyn_global_k_corruptions or ablated_root_differs
                ):
                    raise RuntimeError(
                        "hamlyn ablated condition has no explicit geometry override and uses the "
                        "same root as the aware condition. Pass --hamlyn-ablated-extra-args, "
                        "--prepare-hamlyn-global-k-corruptions, or a distinct "
                        "--hamlyn-ablated-corruptions-root."
                    )
                if dataset == "c3vd" and (
                    dataset == "c3vd"
                    and root == condition_corruptions_root(args, "c3vd", "aware")
                ):
                    raise RuntimeError(
                        f"{dataset} ablated condition has no explicit geometry override. "
                        f"Pass --{dataset}-ablated-extra-args with the repo flag(s), "
                        "or --allow-empty-ablation-overrides if the ablated root already "
                        "contains the wrong-geometry protocol."
                    )

            for model in args.models:
                config = DEFAULT_CONFIGS[dataset][model]
                plan_dir = output_dir / "intermediate" / dataset / condition / model
                output_csv, output_flags = output_csv_for(plan_dir, config)
                command = [
                    args.python,
                    config.script,
                    "--corruptions_root",
                    str(root),
                    "--load_weights_folder",
                    str(config.weights),
                    *config.default_args,
                    *extra_args,
                    *output_flags,
                ]
                warnings = []
                if (
                    condition == "ablated"
                    and not extra_args
                    and root == condition_corruptions_root(args, dataset, "aware")
                ):
                    warnings.append("No condition-specific extra args were supplied.")
                plans.append(
                    RunPlan(
                        dataset=dataset,
                        model=model,
                        condition=condition,
                        repo_root=config.repo_root,
                        command=command,
                        output_csv=output_csv,
                        log_path=plan_dir / "eval.log",
                        manifest_path=plan_dir / "command.json",
                        warnings=warnings,
                    )
                )
    return plans


def print_dry_run(args: argparse.Namespace, plans: list[RunPlan]) -> None:
    print("Geometry ablation dry-run")
    print(f"output_dir: {Path(args.output_dir).expanduser()}")
    print(f"datasets: {', '.join(selected_datasets(args.dataset))}")
    print(f"models: {', '.join(args.models)}")
    print(
        "corruptions: "
        + ("all" if selected_corruptions(args.corruptions) is None else ", ".join(args.corruptions))
    )
    print(f"severities: {', '.join(str(s) for s in args.severities)}")
    for plan in plans:
        print("\n---")
        print(f"dataset: {plan.dataset}")
        print(f"model: {plan.model}")
        print(f"condition: {plan.condition}")
        print(f"workdir: {plan.repo_root}")
        print(f"metrics_csv: {plan.output_csv}")
        print(f"log: {plan.log_path}")
        for warning in plan.warnings:
            print(f"WARNING: {warning}")
        print("command:")
        print(" ".join(shlex.quote(part) for part in plan.command))


def run_plan(plan: RunPlan, args: argparse.Namespace) -> None:
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = plan.command
    removed_args: list[str] = []
    if not args.no_filter_unsupported_args:
        command, removed_args = filter_unsupported_args(plan.command, plan.repo_root)

    with open(plan.manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": plan.dataset,
                "model": plan.model,
                "condition": plan.condition,
                "repo_root": str(plan.repo_root),
                "command": command,
                "original_command": plan.command,
                "removed_unsupported_args": removed_args,
                "output_csv": str(plan.output_csv),
                "warnings": plan.warnings,
            },
            f,
            indent=2,
        )

    if args.skip_existing and plan.output_csv.is_file():
        print(f"[SKIP] {plan.dataset}/{plan.condition}/{plan.model}: {plan.output_csv}")
        return

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"[RUN] {plan.dataset}/{plan.condition}/{plan.model}")
    with open(plan.log_path, "w", encoding="utf-8") as log:
        if removed_args:
            log.write("[INFO] removed unsupported args: " + " ".join(removed_args) + "\n")
        log.write(" ".join(shlex.quote(part) for part in command) + "\n\n")
        proc = subprocess.run(
            command,
            cwd=str(plan.repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for {plan.dataset}/{plan.condition}/{plan.model}. "
            f"See log: {plan.log_path}"
        )
    if not plan.output_csv.is_file():
        raise FileNotFoundError(
            f"Expected metrics CSV was not produced for {plan.dataset}/{plan.condition}/{plan.model}: "
            f"{plan.output_csv}"
        )


def evaluator_supported_options(command: list[str], cwd: Path) -> set[str]:
    help_command = [command[0], command[1], "--help"]
    proc = subprocess.run(
        help_command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = proc.stdout or ""
    return set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", text))


def filter_unsupported_args(command: list[str], cwd: Path) -> tuple[list[str], list[str]]:
    if len(command) < 2:
        return command, []
    supported = evaluator_supported_options(command, cwd)
    if not supported:
        return command, []

    filtered = command[:2]
    removed: list[str] = []
    i = 2
    while i < len(command):
        token = command[i]
        if token.startswith("--") and token not in supported:
            removed.append(token)
            i += 1
            while i < len(command) and not command[i].startswith("--"):
                removed.append(command[i])
                i += 1
            continue
        filtered.append(token)
        i += 1
    return filtered, removed


def first_file_under(root: Path, name: str) -> Path:
    for current_root, _, files in os.walk(root):
        if name in files:
            return Path(current_root) / name
    raise FileNotFoundError(f"Could not find {name} under {root}")


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def prepare_hamlyn_global_k_corruptions(args: argparse.Namespace) -> None:
    aware_root = Path(args.hamlyn_aware_corruptions_root).expanduser()
    output_root = Path(args.hamlyn_ablated_corruptions_root).expanduser()
    if output_root.is_dir() and any(output_root.iterdir()):
        print(f"[INFO] Hamlyn global-K ablated corruptions already exist: {output_root}")
        return
    if not aware_root.is_dir():
        raise FileNotFoundError(f"Hamlyn aware corruptions root not found: {aware_root}")

    if args.hamlyn_global_k_source:
        k0_path = Path(args.hamlyn_global_k_source).expanduser()
    else:
        k0_path = first_file_under(aware_root, "intrinsics.txt")
    if not k0_path.is_file():
        raise FileNotFoundError(f"Hamlyn global K_0 source not found: {k0_path}")
    k0_text = k0_path.read_text(encoding="utf-8")

    log_dir = Path(args.output_dir).expanduser().resolve() / "intermediate" / "hamlyn" / "prepare_global_k"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "manifest.json"
    manifest = {
        "aware_root": str(aware_root),
        "output_root": str(output_root),
        "global_k_source": str(k0_path),
        "file_mode": args.hamlyn_global_k_file_mode,
        "replaced_intrinsics_files": 0,
        "linked_or_copied_files": 0,
    }

    print(f"[RUN] preparing Hamlyn global-K ablated corruptions -> {output_root}")
    print(f"[INFO] Hamlyn K_0 source: {k0_path}")
    for current_root, dirs, files in os.walk(aware_root):
        dirs.sort()
        files.sort()
        current = Path(current_root)
        rel_dir = current.relative_to(aware_root)
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            src = current / filename
            dst = out_dir / filename
            if filename == "intrinsics.txt":
                if not dst.exists():
                    dst.write_text(k0_text, encoding="utf-8")
                manifest["replaced_intrinsics_files"] += 1
            else:
                link_or_copy_file(src, dst, args.hamlyn_global_k_file_mode)
                manifest["linked_or_copied_files"] += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Hamlyn global-K manifest: {manifest_path}")


def prepare_c3vd_raw_corruptions(args: argparse.Namespace) -> None:
    output_root = Path(args.c3vd_ablated_corruptions_root).expanduser()
    if output_root.is_dir() and any(output_root.iterdir()):
        print(f"[INFO] C3VD raw ablated corruptions already exist: {output_root}")
        return

    script = Path(__file__).resolve().parents[1] / "generate_corruptions_from_testlist.py"
    corruptions_arg = (
        "all"
        if selected_corruptions(args.corruptions) is None
        else ",".join(args.corruptions)
    )
    severities_arg = ",".join(str(sev) for sev in args.severities)
    command = [
        args.python,
        str(script),
        "--test_list",
        str(Path(args.c3vd_split_file).expanduser()),
        "--input_root",
        str(Path(args.c3vd_raw_data_root).expanduser()),
        "--output_root",
        str(output_root),
        "--split_name",
        "test",
        "--corruptions",
        corruptions_arg,
        "--severities",
        severities_arg,
    ]

    log_dir = Path(args.output_dir).expanduser().resolve() / "intermediate" / "c3vd" / "prepare_raw"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "generate_corruptions_from_testlist.log"
    manifest_path = log_dir / "command.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_data_root": str(Path(args.c3vd_raw_data_root).expanduser()),
                "output_root": str(output_root),
                "command": command,
            },
            f,
            indent=2,
        )

    print(f"[RUN] preparing C3VD raw ablated corruptions -> {output_root}")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(shlex.quote(part) for part in command) + "\n\n")
        proc = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate C3VD raw corruptions. See log: {log_path}")


def header_map(headers: Iterable[str]) -> dict[str, str]:
    out = {}
    for header in headers:
        key = normalize_text(header).replace("__", "_")
        metric = METRIC_ALIASES.get(key, key)
        out[metric] = header
    return out


def parse_severity(value: str) -> int:
    value = str(value).strip().lower()
    if value.startswith("severity_"):
        value = value.split("_")[-1]
    return int(float(value))


def row_float(row: dict[str, str], hmap: dict[str, str], key: str) -> float:
    raw = row.get(hmap[key], "")
    if raw == "":
        raise ValueError(f"Missing metric {key}")
    return float(raw)


def find_column(hmap: dict[str, str], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        key = normalize_text(candidate)
        if key in hmap:
            return hmap[key]
    return None


def load_metrics(plan: RunPlan, args: argparse.Namespace) -> list[dict]:
    if not plan.output_csv.is_file():
        raise FileNotFoundError(f"Missing CSV for {plan.dataset}/{plan.condition}/{plan.model}: {plan.output_csv}")

    wanted_corruptions = selected_corruptions(args.corruptions)
    wanted_severities = set(args.severities)
    rows = []
    with open(plan.output_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        hmap = header_map(reader.fieldnames)
        corruption_col = find_column(hmap, "corruption", "corr", "corruption_type")
        severity_col = find_column(hmap, "severity", "sev")
        n_col = find_column(hmap, "n_samples", "samples", "n")
        missing_metrics = [metric for metric in ALL_METRICS if metric not in hmap]
        if corruption_col is None or severity_col is None or missing_metrics:
            raise ValueError(
                f"{plan.output_csv} does not look like a severity metrics CSV. "
                f"Missing columns: corruption={corruption_col is None}, "
                f"severity={severity_col is None}, metrics={missing_metrics}"
            )
        for row in reader:
            corruption = normalize_text(row[corruption_col])
            severity = parse_severity(row[severity_col])
            if wanted_corruptions is not None and corruption not in wanted_corruptions:
                continue
            if severity not in wanted_severities:
                continue
            item = {
                "dataset": plan.dataset,
                "model": plan.model,
                "condition": plan.condition,
                "corruption": corruption,
                "severity": severity,
                "source_csv": str(plan.output_csv),
                "n_samples": int(float(row[n_col])) if n_col and row.get(n_col, "") else "",
            }
            for metric in ALL_METRICS:
                item[metric] = row_float(row, hmap, metric)
            item["mders"] = compute_mders(item)
            rows.append(item)
    return rows


def compute_mders(values: dict[str, float]) -> float:
    accuracy = (values["a1"] + values["a2"] + values["a3"]) / 3.0
    error = (
        values["abs_rel"]
        + values["sq_rel"]
        + values["rmse"]
        + values["rmse_log"]
    ) / 4.0
    return accuracy / (1.0 + error)


def average_metrics(rows: list[dict]) -> dict[str, float]:
    out = {metric: mean(float(row[metric]) for row in rows) for metric in ALL_METRICS}
    out["mders"] = compute_mders(out)
    return out


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def mean_by_corruption(rows: list[dict], metric: str) -> dict[str, float]:
    grouped = group_rows(rows, ("corruption",))
    return {
        key[0]: mean(float(row[metric]) for row in bucket)
        for key, bucket in grouped.items()
    }


def compute_mce_for_model(
    model_rows: list[dict],
    baseline_rows: list[dict],
    metric: str,
    lower_is_better: bool,
) -> float:
    model_by_corr = mean_by_corruption(model_rows, metric)
    base_by_corr = mean_by_corruption(baseline_rows, metric)
    common = sorted(set(model_by_corr).intersection(base_by_corr))
    if not common:
        raise ValueError(f"No common corruptions for mCE metric {metric}")
    ratios = []
    for corr in common:
        model_value = model_by_corr[corr]
        base_value = base_by_corr[corr]
        if model_value == 0 or base_value == 0:
            continue
        ratio = model_value / base_value if lower_is_better else base_value / model_value
        ratios.append(ratio)
    if not ratios:
        raise ValueError(f"No non-zero ratios for mCE metric {metric}")
    return 100.0 * mean(ratios)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = group_rows(rows, ("dataset", "condition", "model"))
    baselines = {
        (dataset, condition): bucket
        for (dataset, condition, model), bucket in grouped.items()
        if model == "monodepth2"
    }
    summaries = []
    for (dataset, condition, model), bucket in sorted(grouped.items()):
        baseline = baselines.get((dataset, condition))
        if baseline is None:
            raise ValueError(f"Missing Monodepth2 baseline for {dataset}/{condition}")
        avg = average_metrics(bucket)
        item = {
            "dataset": dataset,
            "model": model,
            "condition": condition,
            **avg,
        }
        for metric in ERROR_METRICS:
            item[f"mce_{metric}"] = compute_mce_for_model(
                bucket, baseline, metric, lower_is_better=True
            )
        for metric in ACCURACY_METRICS:
            item[f"mce_{metric}"] = compute_mce_for_model(
                bucket, baseline, metric, lower_is_better=False
            )
        item["mean_error_mce"] = mean(item[f"mce_{m}"] for m in ERROR_METRICS)
        item["mean_accuracy_mce"] = mean(item[f"mce_{m}"] for m in ACCURACY_METRICS)
        item["mean_mce"] = mean(item[f"mce_{m}"] for m in ALL_METRICS)
        summaries.append(item)
    return summaries


def delta_table(summaries: list[dict]) -> list[dict]:
    by_key = {
        (row["dataset"], row["model"], row["condition"]): row
        for row in summaries
    }
    out = []
    for dataset in DATASETS:
        models = sorted({row["model"] for row in summaries if row["dataset"] == dataset})
        for model in models:
            aware = by_key.get((dataset, model, "aware"))
            ablated = by_key.get((dataset, model, "ablated"))
            if aware is None or ablated is None:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mce_aware": aware["mean_mce"],
                    "mce_ablated": ablated["mean_mce"],
                    "delta_mce": ablated["mean_mce"] - aware["mean_mce"],
                    "mders_aware": aware["mders"],
                    "mders_ablated": ablated["mders"],
                    "delta_mders": ablated["mders"] - aware["mders"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float) -> str:
    return f"{float(value):.3f}"


def write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Model & Geometry-aware mCE $\downarrow$ & Ablated mCE $\downarrow$ & $\Delta$mCE $\uparrow$ & Geometry-aware mDERS $\uparrow$ & Ablated mDERS $\uparrow$ & $\Delta$mDERS $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["dataset"].upper(),
                    row["model"],
                    format_float(row["mce_aware"]),
                    format_float(row["mce_ablated"]),
                    format_float(row["delta_mce"]),
                    format_float(row["mders_aware"]),
                    format_float(row["mders_ablated"]),
                    format_float(row["delta_mders"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plans = build_plan(args)
    if args.dry_run:
        print_dry_run(args, plans)
        if args.prepare_hamlyn_global_k_corruptions:
            print("\nHamlyn global-K corruption preparation would run before evaluation:")
            print(f"  aware_root: {Path(args.hamlyn_aware_corruptions_root).expanduser()}")
            print(f"  output_root: {Path(args.hamlyn_ablated_corruptions_root).expanduser()}")
            print(
                "  global_k_source: "
                + str(Path(args.hamlyn_global_k_source).expanduser())
                if args.hamlyn_global_k_source
                else "  global_k_source: first intrinsics.txt found under aware_root"
            )
        if args.prepare_c3vd_raw_corruptions:
            print("\nC3VD raw corruption preparation would run before evaluation:")
            print(f"  raw_data_root: {Path(args.c3vd_raw_data_root).expanduser()}")
            print(f"  output_root: {Path(args.c3vd_ablated_corruptions_root).expanduser()}")
            print(f"  split_file: {Path(args.c3vd_split_file).expanduser()}")
        return

    if args.prepare_hamlyn_global_k_corruptions and "hamlyn" in selected_datasets(args.dataset):
        prepare_hamlyn_global_k_corruptions(args)

    if args.prepare_c3vd_raw_corruptions and "c3vd" in selected_datasets(args.dataset):
        prepare_c3vd_raw_corruptions(args)

    if not args.collect_only:
        for plan in plans:
            run_plan(plan, args)

    raw_rows = []
    for plan in plans:
        raw_rows.extend(load_metrics(plan, args))
    if not raw_rows:
        raise RuntimeError("No raw metrics were collected.")

    summary_rows = summarize(raw_rows)
    delta_rows = delta_table(summary_rows)

    raw_fields = [
        "dataset",
        "model",
        "condition",
        "corruption",
        "severity",
        *ALL_METRICS,
        "mders",
        "n_samples",
        "source_csv",
    ]
    summary_fields = [
        "dataset",
        "model",
        "condition",
        *ALL_METRICS,
        "mders",
        *(f"mce_{m}" for m in ALL_METRICS),
        "mean_error_mce",
        "mean_accuracy_mce",
        "mean_mce",
    ]
    delta_fields = [
        "dataset",
        "model",
        "mce_aware",
        "mce_ablated",
        "delta_mce",
        "mders_aware",
        "mders_ablated",
        "delta_mders",
    ]

    write_csv(output_dir / "geometry_ablation_raw_metrics.csv", raw_rows, list(raw_fields))
    write_csv(output_dir / "geometry_ablation_summary.csv", summary_rows, list(summary_fields))
    write_csv(output_dir / "geometry_ablation_delta.csv", delta_rows, list(delta_fields))
    write_latex(output_dir / "geometry_ablation_table.tex", delta_rows)
    print(f"Saved raw metrics: {output_dir / 'geometry_ablation_raw_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'geometry_ablation_summary.csv'}")
    print(f"Saved delta: {output_dir / 'geometry_ablation_delta.csv'}")
    print(f"Saved LaTeX table: {output_dir / 'geometry_ablation_table.tex'}")


if __name__ == "__main__":
    main()
