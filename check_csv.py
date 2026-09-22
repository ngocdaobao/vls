import pandas as pd
import csv 

"""Find the tasks of an evaluation run that need to be re-run.

A task is broken when its ``Task_<id>`` directory is missing, empty, holds no
``.mp4`` rollout, or carries an ``error*`` marker. Broken directories are
deleted (unless ``cleanup=False``) so the re-run starts from a clean slate.

Used both as a CLI and as a library: main.py imports find_error_tasks() to build
the task selection for a re-run (see main.rerun_error_dir).
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

# Suite sizes come from LIBERO-plus' task_classification.json (one entry per
# perturbation variant), so the scan covers exactly the ids the suite defines.
_TASK_CLASSIFICATION_FILE = (
    Path(__file__).parent / "third_party" / "libero_plus" / "libero" / "libero"
    / "benchmark" / "task_classification.json"
)


def suite_task_count(suite_name: str) -> int:
    """Number of task variants in a LIBERO-plus suite (ids are 0..count-1)."""
    with _TASK_CLASSIFICATION_FILE.open() as f:
        classification = json.load(f)
    if suite_name not in classification:
        raise ValueError(
            f"Unknown suite '{suite_name}'. Available: {', '.join(sorted(classification))}"
        )
    return len(classification[suite_name])


def find_error_tasks(
    file_path: str,
    num_tasks: Optional[int] = None,
    suite_name: Optional[str] = None,
    task_ids: Optional[Iterable[int]] = None,
    cleanup: bool = True,
) -> List[int]:
    """Return the sorted ids of the tasks that did not produce a valid rollout.

    Args:
        file_path: Directory holding the per-task output dirs (``Task_<id>``),
            i.e. ``outputs/<suite_name>/``.
        num_tasks: Scan ids ``0..num_tasks-1``. Defaults to the suite's size.
        suite_name: Suite whose task count bounds the scan; defaults to the name
            of the output directory, which main.py names after the suite.
        task_ids: Scan only these ids, instead of the whole suite.
        cleanup: Delete the broken task directories, so the re-run does not
            resume from a half-written episode. Missing dirs are left alone.
    """
    if task_ids is not None:
        candidates = [int(task_id) for task_id in task_ids]
    else:
        if num_tasks is None:
            suite = suite_name or os.path.basename(os.path.normpath(file_path))
            num_tasks = suite_task_count(suite)
        candidates = list(range(num_tasks))
    df = pd.read_csv(file_path)
    complete = df['Task ID'].unique().tolist()
    error_task = []

    for i in candidates:
        if i not in complete:
            error_task.append(i)

    return sorted(set(error_task))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check file.")
    parser.add_argument("--file_path", type=str, help="Path to the file to check.")
    parser.add_argument("--suite_name", type=str, default=None,
                        help="Suite whose task count bounds the scan "
                             "(default: the name of the output directory).")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Explicit id range to scan (0..N-1), overriding the suite size.")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Report the broken tasks without deleting their directories.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the error task ids to this file as JSON.")
    args = parser.parse_args()

    error_task = find_error_tasks(
        args.file_path,
        num_tasks=args.num_tasks,
        suite_name=args.suite_name,
        cleanup=not args.no_cleanup,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(error_task, f)
        print("Wrote error tasks to: ", args.output)

    print("Error tasks: ", error_task)
    print("Total error tasks: ", len(error_task))
    print(f"{error_task[len(error_task)//2] if error_task else 'N/A'}")

