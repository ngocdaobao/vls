"""Summarise an evaluation run into ``result/<suite_name>.csv``.

One row per task that produced a rollout video (``Task_<id>/*.mp4``), with its
status and LIBERO-plus perturbation category. main.py runs only the tasks that
are not listed yet (`pending_task_ids`) and appends a row as each task finishes
(`append_result`), so the CLI below is only needed to rebuild the CSV from an
output directory.
"""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

# One entry per task variant; entry ``id`` is 1-based and lines up with the
# suite's task index (Task ID = id - 1), the order libero_task_map defines.
_TASK_CLASSIFICATION_FILE = (
    Path(__file__).parent / "third_party" / "libero_plus" / "libero" / "libero"
    / "benchmark" / "task_classification.json"
)

CSV_COLUMNS = ["Task ID", "Status", "Category"]


def load_task_categories(suite_name: str) -> Dict[int, Optional[str]]:
    """Task ID -> perturbation category for a LIBERO-plus suite."""
    with _TASK_CLASSIFICATION_FILE.open() as f:
        classification = json.load(f)
    if suite_name not in classification:
        raise ValueError(
            f"Unknown suite '{suite_name}'. Available: {', '.join(sorted(classification))}"
        )
    return {int(entry["id"]) - 1: entry.get("category") for entry in classification[suite_name]}


def suite_task_count(suite_name: str) -> int:
    """Number of task variants in a LIBERO-plus suite (ids are 0..count-1)."""
    return len(load_task_categories(suite_name))


def collect_results(output_dir: str) -> Dict[int, str]:
    """Task ID -> "success"/"fail" for every ``Task_<id>`` dir holding an .mp4."""
    result = {}
    for task in os.listdir(output_dir):
        task_dir = os.path.join(output_dir, task)
        if not task.startswith("Task_") or not os.path.isdir(task_dir):
            continue
        videos = [file for file in os.listdir(task_dir) if file.endswith(".mp4")]
        if videos:
            task_id = int(task.split("_")[-1])
            result[task_id] = "success" if any("success" in v for v in videos) else "fail"
    return result


def completed_task_ids(csv_path: str) -> Set[int]:
    """Task IDs already listed in a result CSV (empty if the file does not exist)."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["Task ID"]) for row in csv.DictReader(f) if row.get("Task ID", "").strip()}


def pending_task_ids(
    csv_path: str,
    suite_name: str,
    task_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """Sorted task IDs (of ``task_ids``, default the whole suite) missing from the CSV."""
    candidates = range(suite_task_count(suite_name)) if task_ids is None else task_ids
    done = completed_task_ids(csv_path)
    return sorted({int(task_id) for task_id in candidates} - done)


def append_result(csv_path: str, task_id: int, status: str, category: Optional[str]) -> None:
    """Append one task's row, writing the header first if the file is new or empty.

    Several multi-GPU workers append to the same file, so the whole
    check-header-then-write is done under an exclusive lock.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([int(task_id), status, category])
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Write the per-task results of a LIBERO-plus run to CSV.")
    arg_parser.add_argument("--result-dir", type=str, default=None,
                            help="Run output dir holding Task_<id> dirs (default: outputs/<suite>).")
    arg_parser.add_argument("--suite", type=str, default=None,
                            help="Suite name (default: name of --result-dir).")
    arg_parser.add_argument("--output", type=str, default=None,
                            help="CSV path (default: result/<suite>.csv).")
    args = arg_parser.parse_args()

    if args.suite is None and args.result_dir is None:
        arg_parser.error("pass --suite or --result-dir")
    suite = args.suite or os.path.basename(os.path.normpath(args.result_dir))
    result_dir = args.result_dir or os.path.join("outputs", suite)
    output = args.output or os.path.join("result", f"{suite}.csv")

    categories = load_task_categories(suite)
    result = collect_results(result_dir)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for task_id in sorted(result):
            writer.writerow([task_id, result[task_id], categories.get(task_id)])
    print(f"Wrote {len(result)} task(s) from '{result_dir}' to '{output}'.")
