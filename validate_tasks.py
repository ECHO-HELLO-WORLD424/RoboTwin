#!/usr/bin/env python3
"""
Validate composed OOD tasks by running play_once() with a timeout.

For each task and seed, this script:
  1. Initialises the SAPIEN environment
  2. Runs the expert planner (play_once) inside a child process
  3. Kills the child if it exceeds the timeout (catches infinite loops)
  4. Reports plan_success, check_success, and any errors

Usage:
    python validate_tasks.py [--tasks TASK ...] [--seeds N] [--timeout SEC]
"""

import sys
import os
import importlib
import argparse
import traceback
import signal
import multiprocessing as mp
from pathlib import Path

import yaml
import numpy as np

# ── bootstrap paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from description.utils.generate_episode_instructions import generate_episode_descriptions


# ── helpers ──────────────────────────────────────────────────────────
def load_task_config(task_config_name: str = "demo_clean") -> dict:
    with open(f"./task_config/{task_config_name}.yml", "r") as f:
        return yaml.safe_load(f)


def get_embodiment_file(embodiment_type: str, embodiment_types: dict) -> str:
    return embodiment_types[embodiment_type]["file_path"]


def get_embodiment_config(robot_file: str) -> dict:
    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        return yaml.safe_load(f)


def build_args(task_name: str, task_config_name: str = "demo_clean") -> dict:
    """Build the args dict that _init_task_env_ expects."""
    args = load_task_config(task_config_name)
    args["task_name"] = task_name
    args["task_config"] = task_config_name
    args["ckpt_setting"] = "0"
    args["save_root"] = "./validate_results"

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.safe_load(f)

    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0], _embodiment_types)
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1], _embodiment_types)
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    with open(CONFIGS_PATH + "_camera_config.yml", "r") as f:
        _camera_config = yaml.safe_load(f)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    # Disable rendering and video logging for validation
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["collect_data"] = False
    args["eval_mode"] = True

    return args


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


# ── worker that runs in a child process ──────────────────────────────
def _run_single_seed(task_name, args, seed, result_queue):
    """Run setup_demo + play_once + check_success for one seed.
    Puts a result dict onto result_queue."""
    result = {
        "seed": seed,
        "setup_ok": False,
        "play_ok": False,
        "plan_success": False,
        "check_success": False,
        "instruction_ok": False,
        "error": None,
    }
    TASK_ENV = None
    try:
        TASK_ENV = class_decorator(task_name)
        TASK_ENV.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        result["setup_ok"] = True

        episode_info = TASK_ENV.play_once()
        result["play_ok"] = True
        result["plan_success"] = bool(TASK_ENV.plan_success)

        if TASK_ENV.plan_success:
            result["check_success"] = bool(TASK_ENV.check_success())

        # Validate instruction generation
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, 10)
        if results and len(results) > 0 and len(results[0].get("seen", [])) > 0:
            result["instruction_ok"] = True
        else:
            result["error"] = "No valid instructions generated"

    except UnStableError:
        result["error"] = "UnStableError (physics instability)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        if TASK_ENV is not None:
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
    result_queue.put(result)


def run_with_timeout(task_name, args, seed, timeout):
    """Run a single seed in a child process with a timeout.
    Returns a result dict."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_single_seed, args=(task_name, args, seed, q))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)
        return {
            "seed": seed,
            "setup_ok": False,
            "play_ok": False,
            "plan_success": False,
            "check_success": False,
            "instruction_ok": False,
            "error": f"TIMEOUT after {timeout}s (possible infinite loop in planner)",
        }

    if q.empty():
        return {
            "seed": seed,
            "setup_ok": False,
            "play_ok": False,
            "plan_success": False,
            "check_success": False,
            "instruction_ok": False,
            "error": "Process exited without result (crash or segfault)",
        }

    return q.get_nowait()


# ── main ─────────────────────────────────────────────────────────────
DEFAULT_TASKS = [
    "handover_then_hang_mug",
    "click_bell_then_sort_blocks",
    "stamp_then_stack_bowls",
]


def main():
    parser = argparse.ArgumentParser(description="Validate composed OOD tasks")
    parser.add_argument(
        "--tasks", nargs="+", default=DEFAULT_TASKS,
        help="Task names to validate",
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of seeds to test per task (starting from 10000)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout in seconds per seed (kills process if exceeded)",
    )
    parser.add_argument(
        "--task-config", type=str, default="demo_clean",
        help="Task config name",
    )
    cli_args = parser.parse_args()

    # Suppress rendering
    os.environ.setdefault("DISPLAY", "")

    print("=" * 70)
    print("  OOD Task Validator")
    print("=" * 70)
    print(f"  Tasks:   {cli_args.tasks}")
    print(f"  Seeds:   {cli_args.seeds} (starting from 10000)")
    print(f"  Timeout: {cli_args.timeout}s per seed")
    print("=" * 70)

    all_passed = True

    for task_name in cli_args.tasks:
        print(f"\n{'─' * 60}")
        print(f"  Task: {task_name}")
        print(f"{'─' * 60}")

        args = build_args(task_name, cli_args.task_config)

        stats = {
            "total": 0,
            "setup_ok": 0,
            "play_ok": 0,
            "plan_success": 0,
            "check_success": 0,
            "instruction_ok": 0,
            "timeout": 0,
            "error": 0,
        }

        for i in range(cli_args.seeds):
            seed = 10000 + i
            stats["total"] += 1

            result = run_with_timeout(task_name, args, seed, cli_args.timeout)

            status_icon = "✓" if result["play_ok"] and result["plan_success"] else "✗"
            if "TIMEOUT" in (result["error"] or ""):
                status_icon = "⏱"
                stats["timeout"] += 1

            if result["setup_ok"]:
                stats["setup_ok"] += 1
            if result["play_ok"]:
                stats["play_ok"] += 1
            if result["plan_success"]:
                stats["plan_success"] += 1
            if result["check_success"]:
                stats["check_success"] += 1
            if result["instruction_ok"]:
                stats["instruction_ok"] += 1
            if result["error"] and "TIMEOUT" not in result["error"]:
                stats["error"] += 1

            err_msg = ""
            if result["error"]:
                # Show first line of error only
                err_msg = f"  | {result['error'].split(chr(10))[0]}"

            print(
                f"  {status_icon} seed={seed:5d}  "
                f"setup={result['setup_ok']}  "
                f"play={result['play_ok']}  "
                f"plan={result['plan_success']}  "
                f"succ={result['check_success']}  "
                f"instr={result['instruction_ok']}"
                f"{err_msg}"
            )

        # Summary
        print(f"\n  Summary for {task_name}:")
        print(f"    Setup OK:       {stats['setup_ok']}/{stats['total']}")
        print(f"    Play OK:        {stats['play_ok']}/{stats['total']}")
        print(f"    Plan success:   {stats['plan_success']}/{stats['total']}")
        print(f"    Check success:  {stats['check_success']}/{stats['total']}")
        print(f"    Instruction OK: {stats['instruction_ok']}/{stats['total']}")
        print(f"    Timeouts:       {stats['timeout']}/{stats['total']}")
        print(f"    Errors:         {stats['error']}/{stats['total']}")

        # Fail criteria: >50% timeout or >80% errors or 0 plan_success
        fail = False
        if stats["timeout"] > stats["total"] * 0.5:
            print(f"\n  FAIL: Too many timeouts ({stats['timeout']}/{stats['total']})")
            fail = True
        if stats["plan_success"] == 0:
            print(f"\n  FAIL: No seeds achieved plan_success")
            fail = True
        if stats["instruction_ok"] == 0 and stats["play_ok"] > 0:
            print(f"\n  FAIL: Instruction generation broken for all seeds")
            fail = True

        if fail:
            all_passed = False
        else:
            print(f"\n  PASS")

    print(f"\n{'=' * 70}")
    if all_passed:
        print("  ALL TASKS PASSED")
    else:
        print("  SOME TASKS FAILED")
        sys.exit(1)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
