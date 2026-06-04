#!/usr/bin/env python3
"""
Launcher unico per training e render.

Supporta due motori del repo:
- host  -> HoST / Isaac Gym
- mjlab -> unitree_rl_mjlab / MuJoCo

Uso rapido:
  python3 TRAIN_RENDER.py
  python3 TRAIN_RENDER.py --engine mjlab --task Unitree-G1-Flat --mode both
  python3 TRAIN_RENDER.py --engine host --task g1_ground --mode train
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
HARDWARE_CONFIG = ROOT_DIR / "hardware_config.yaml"

HOST_DIR = ROOT_DIR / "HoST"
HOST_LEGGED_GYM_DIR = HOST_DIR / "legged_gym"
HOST_SCRIPTS_DIR = HOST_LEGGED_GYM_DIR / "legged_gym" / "scripts"
HOST_LOGS_DIR = HOST_LEGGED_GYM_DIR / "logs"

MJLAB_DIR = ROOT_DIR / "unitree_rl_mjlab"
MJLAB_SCRIPTS_DIR = MJLAB_DIR / "scripts"
MJLAB_LOGS_DIR = MJLAB_DIR / "logs" / "rsl_rl"

HOST_TASKS = (
    "g1_ground",
    "g1_platform",
    "g1_wall",
    "g1_slope",
    "g1_ground_prone",
    "h1_ground",
    "pi_ground",
)

MJLAB_TASKS = (
    "Unitree-G1-GetUp",
    "Unitree-G1-Flat",
    "Unitree-G1-Rough",
    "Unitree-G1-23Dof-Flat",
    "Unitree-G1-23Dof-Rough",
    "Unitree-G1-Tracking-No-State-Estimation",
    "Unitree-G1-23Dof-Tracking-No-State-Estimation",
    "Unitree-Go2-Flat",
    "Unitree-A2-Flat",
    "Unitree-R1-Flat",
    "Unitree-H1_2-Flat",
)

DEFAULT_CONFIG = {
    "engine": "auto",
    "mode": "both",
    "task": "Unitree-G1-Flat",
    "experiment_name": None,
    "run_name": "train_render_quick",
    "logger": None,
    "max_iterations": 100,
    "num_envs": None,
    "seed": 1,
    "rl_device": None,
    "sim_device": None,
    "device": None,
    "graphics_device_id": 0,
    "checkpoint": -1,
    "checkpoint_file": None,
    "resume": False,
    "load_run": None,
    "viewer": "auto",
    "headless_train": True,
    "headless_play": False,
    "play_num_envs": 1,
    "extra_train_args": [],
    "extra_play_args": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper unico per training e render HumanUP."
    )
    parser.add_argument("--engine", choices=("auto", "host", "mjlab"), default=None)
    parser.add_argument("--mode", choices=("train", "play", "both"), default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rl-device", default=None)
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graphics-device-id", type=int, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--load-run", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default=None)
    parser.add_argument("--headless-train", action="store_true")
    parser.add_argument("--viewer-train", action="store_true")
    parser.add_argument("--headless-play", action="store_true")
    parser.add_argument("--viewer-play", action="store_true")
    parser.add_argument("--play-num-envs", type=int, default=None)
    parser.add_argument("--extra-train-arg", action="append", default=[])
    parser.add_argument("--extra-play-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def read_hardware_config() -> dict:
    config = {
        "backend": "cpu",
        "cpu": {
            "num_envs": 64,
            "mujoco_gl": "osmesa",
            "torch_device": "cpu",
            "num_threads": 8,
        },
        "cuda": {
            "gpu_ids": [0],
            "num_envs": 4096,
            "mujoco_gl": "egl",
            "torch_device": "cuda",
        },
    }
    if not HARDWARE_CONFIG.exists():
        return config

    section = None
    line_re = re.compile(r"^([A-Za-z0-9_]+):\s*(.+)?$")
    for raw_line in HARDWARE_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        if raw_line and not raw_line[0].isspace():
            match = line_re.match(line)
            if not match:
                continue
            key, value = match.groups()
            if value is None or value == "":
                section = key
            else:
                section = None
                config[key] = parse_yaml_scalar(value)
            continue

        if section is None:
            continue
        stripped = line.strip()
        match = line_re.match(stripped)
        if not match:
            continue
        key, value = match.groups()
        if value is not None:
            config.setdefault(section, {})[key] = parse_yaml_scalar(value)

    return config


def parse_yaml_scalar(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        parsed = []
        for item in items:
            try:
                parsed.append(int(item))
            except ValueError:
                parsed.append(item.strip("'\""))
        return parsed
    if value.isdigit():
        return int(value)
    return value


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def infer_engine(task: str, requested_engine: str, isaacgym_ok: bool) -> str:
    if requested_engine != "auto":
        return requested_engine
    if task in HOST_TASKS:
        return "host"
    if task in MJLAB_TASKS:
        return "mjlab"
    if isaacgym_ok:
        return "host"
    return "mjlab"


def build_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULT_CONFIG)
    hardware = read_hardware_config()
    backend = hardware.get("backend", "cpu")
    backend_cfg = hardware.get(backend, {})
    train_cfg = hardware.get("train", {})

    if config["num_envs"] is None:
        config["num_envs"] = backend_cfg.get("num_envs", 64)
    if config["logger"] is None:
        config["logger"] = train_cfg.get("logger", "tensorboard")

    if backend == "cpu":
        config["rl_device"] = "cpu"
        config["sim_device"] = "cpu"
        config["device"] = "cpu"
    else:
        gpu_ids = hardware.get("cuda", {}).get("gpu_ids", [0])
        gpu_id = gpu_ids[0] if gpu_ids else 0
        config["rl_device"] = f"cuda:{gpu_id}"
        config["sim_device"] = f"cuda:{gpu_id}"
        config["device"] = f"cuda:{gpu_id}"
        config["graphics_device_id"] = gpu_id

    for key in (
        "engine",
        "mode",
        "task",
        "experiment_name",
        "run_name",
        "logger",
        "max_iterations",
        "num_envs",
        "seed",
        "rl_device",
        "sim_device",
        "device",
        "graphics_device_id",
        "checkpoint",
        "checkpoint_file",
        "load_run",
        "viewer",
        "play_num_envs",
    ):
        cli_value = getattr(args, key)
        if cli_value is not None:
            config[key] = cli_value

    if args.resume:
        config["resume"] = True

    if args.viewer_train:
        config["headless_train"] = False
    elif args.headless_train:
        config["headless_train"] = True

    if args.viewer_play:
        config["headless_play"] = False
    elif args.headless_play:
        config["headless_play"] = True

    if args.extra_train_arg:
        config["extra_train_args"] = args.extra_train_arg
    if args.extra_play_arg:
        config["extra_play_args"] = args.extra_play_arg

    isaacgym_ok = module_available("isaacgym")
    config["isaacgym_available"] = isaacgym_ok
    config["hardware_backend"] = backend
    config["mujoco_gl"] = backend_cfg.get("mujoco_gl", "osmesa")
    config["cpu_threads"] = backend_cfg.get("num_threads", 8)
    config["gpu_ids"] = hardware.get("cuda", {}).get("gpu_ids", [0])
    config["engine"] = infer_engine(config["task"], config["engine"], isaacgym_ok)

    if not config["experiment_name"]:
        config["experiment_name"] = default_experiment_name(config["task"], config["engine"])

    return config


def default_experiment_name(task: str, engine: str) -> str:
    if engine == "host":
        return task
    mapping = {
        "Unitree-G1-GetUp": "g1_getup",
        "Unitree-G1-Flat": "g1_velocity",
        "Unitree-G1-Rough": "g1_velocity",
        "Unitree-G1-23Dof-Flat": "g1_23dof_velocity",
        "Unitree-G1-23Dof-Rough": "g1_23dof_velocity",
        "Unitree-G1-Tracking-No-State-Estimation": "g1_tracking",
        "Unitree-G1-23Dof-Tracking-No-State-Estimation": "g1_23dof_tracking",
    }
    return mapping.get(task, task.lower().replace("unitree-", "").replace("-", "_"))


def ensure_engine_ready(config: dict) -> None:
    if config["engine"] == "host":
        required = [HOST_DIR, HOST_LEGGED_GYM_DIR, HOST_SCRIPTS_DIR]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Percorsi HoST mancanti:\n- " + "\n- ".join(missing))
        if config["task"] not in HOST_TASKS:
            raise ValueError(
                f"Task '{config['task']}' non valido per engine=host.\n"
                f"Task host disponibili: {', '.join(HOST_TASKS)}"
            )
        if not config["isaacgym_available"]:
            raise RuntimeError(
                "Il task richiesto usa HoST/Isaac Gym ma il modulo 'isaacgym' non e' installato "
                f"nell'interprete {sys.executable}.\n"
                "Se vuoi usare HoST devi installare Isaac Gym come indicato in docs/install.md.\n"
                "Se invece vuoi usare il backend CPU/MuJoCo del repo, prova per esempio:\n"
                "python3 TRAIN_RENDER.py --engine mjlab --task Unitree-G1-Flat --mode train"
            )
    else:
        required = [MJLAB_DIR, MJLAB_SCRIPTS_DIR]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Percorsi unitree_rl_mjlab mancanti:\n- " + "\n- ".join(missing))
        if config["task"] not in MJLAB_TASKS:
            raise ValueError(
                f"Task '{config['task']}' non valido per engine=mjlab.\n"
                f"Task mjlab disponibili: {', '.join(MJLAB_TASKS)}"
            )


def build_env(config: dict) -> tuple[dict, Path]:
    env = os.environ.copy()

    if config["engine"] == "host":
        pythonpath_parts = [str(HOST_LEGGED_GYM_DIR), str(HOST_DIR / "rsl_rl")]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        return env, HOST_LEGGED_GYM_DIR

    # For pure headless training, letting MuJoCo use its default import path is
    # more robust than forcing a GL backend before Python starts.
    needs_gl_backend = config["mode"] in ("play", "both") or not config["headless_train"]
    if needs_gl_backend:
        env["MUJOCO_GL"] = str(config["mujoco_gl"])
    else:
        env.pop("MUJOCO_GL", None)
    env["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    if config["hardware_backend"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config["gpu_ids"])
    return env, MJLAB_DIR


def build_host_train_command(config: dict) -> list[str]:
    command = [sys.executable, str(HOST_SCRIPTS_DIR / "train.py")]
    command.extend(["--task", config["task"]])
    command.extend(["--experiment_name", config["experiment_name"]])
    command.extend(["--run_name", config["run_name"]])
    command.extend(["--num_envs", str(config["num_envs"])])
    command.extend(["--seed", str(config["seed"])])
    command.extend(["--rl_device", config["rl_device"]])
    command.extend(["--sim_device", config["sim_device"]])
    command.extend(["--graphics_device_id", str(config["graphics_device_id"])])
    command.extend(["--max_iterations", str(config["max_iterations"])])
    if config["resume"]:
        command.append("--resume")
    if config["load_run"]:
        command.extend(["--load_run", str(config["load_run"])])
    if config["checkpoint"] is not None:
        command.extend(["--checkpoint", str(config["checkpoint"])])
    if config["headless_train"]:
        command.append("--headless")
    command.extend(config["extra_train_args"])
    return command


def build_host_play_command(config: dict, load_run: str | None) -> list[str]:
    command = [sys.executable, str(HOST_SCRIPTS_DIR / "play.py")]
    command.extend(["--task", config["task"]])
    command.extend(["--experiment_name", config["experiment_name"]])
    command.extend(["--run_name", config["run_name"]])
    command.extend(["--num_envs", str(config["play_num_envs"])])
    command.extend(["--seed", str(config["seed"])])
    command.extend(["--rl_device", config["rl_device"]])
    command.extend(["--sim_device", config["sim_device"]])
    command.extend(["--graphics_device_id", str(config["graphics_device_id"])])
    command.append("--resume")
    if load_run:
        command.extend(["--load_run", load_run])
    if config["checkpoint"] is not None:
        command.extend(["--checkpoint", str(config["checkpoint"])])
    if config["headless_play"]:
        command.append("--headless")
    command.extend(config["extra_play_args"])
    return command


def build_mjlab_train_command(config: dict) -> list[str]:
    command = [sys.executable, str(MJLAB_SCRIPTS_DIR / "train.py"), config["task"]]
    command.append(f"--agent.max-iterations={config['max_iterations']}")
    command.append(f"--agent.experiment-name={config['experiment_name']}")
    command.append(f"--agent.run-name={config['run_name']}")
    command.append(f"--agent.seed={config['seed']}")
    command.append(f"--agent.logger={config['logger']}")
    command.append(f"--env.scene.num-envs={config['num_envs']}")
    if config["resume"]:
        command.append("--agent.resume")
    if config["load_run"]:
        command.append(f"--agent.load-run={config['load_run']}")
    if config["checkpoint"] not in (None, -1):
        command.append(f"--agent.load-checkpoint={config['checkpoint']}")
    if config["hardware_backend"] == "cpu":
        command.extend(["--gpu-ids", "None"])
    elif config["gpu_ids"]:
        command.append("--gpu-ids")
        command.extend(str(gpu) for gpu in config["gpu_ids"])
    command.extend(config["extra_train_args"])
    return command


def build_mjlab_play_command(config: dict, checkpoint_file: str) -> list[str]:
    command = [sys.executable, str(MJLAB_SCRIPTS_DIR / "play.py"), config["task"]]
    command.append(f"--checkpoint-file={checkpoint_file}")
    command.append(f"--num-envs={config['play_num_envs']}")
    command.append(f"--device={config['device']}")
    command.append(f"--viewer={config['viewer']}")
    command.extend(config["extra_play_args"])
    return command


def run_command(command: list[str], env: dict, cwd: Path, dry_run: bool = False) -> None:
    print("\n[CMD]")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def find_latest_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Directory non trovata: {root}")
    dirs = [path for path in root.iterdir() if path.is_dir() and path.name != "exported"]
    if not dirs:
        raise FileNotFoundError(f"Nessun run disponibile in {root}")
    dirs.sort(key=lambda path: path.stat().st_mtime)
    return dirs[-1]


def find_latest_host_run(experiment_name: str) -> str:
    return find_latest_dir(HOST_LOGS_DIR / experiment_name).name


def find_latest_mjlab_checkpoint(experiment_name: str, load_run: str | None) -> str:
    experiment_dir = MJLAB_LOGS_DIR / experiment_name
    run_dir = experiment_dir / load_run if load_run else find_latest_dir(experiment_dir)
    models = list(run_dir.glob("model_*.pt"))
    if not models:
        raise FileNotFoundError(f"Nessun checkpoint trovato in {run_dir}")
    models.sort(key=model_sort_key)
    return str(models[-1])


def model_sort_key(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def print_summary(config: dict) -> None:
    print("[TRAIN_RENDER]")
    for key in (
        "engine",
        "mode",
        "task",
        "experiment_name",
        "run_name",
        "logger",
        "max_iterations",
        "num_envs",
        "seed",
        "hardware_backend",
        "rl_device",
        "sim_device",
        "device",
        "graphics_device_id",
        "checkpoint",
        "checkpoint_file",
        "resume",
        "load_run",
        "viewer",
        "headless_train",
        "headless_play",
        "play_num_envs",
        "isaacgym_available",
    ):
        print(f"- {key}: {config.get(key)}")


def print_tasks() -> None:
    print("[HOST TASKS]")
    for task in HOST_TASKS:
        print(f"- {task}")
    print("\n[MJLAB TASKS]")
    for task in MJLAB_TASKS:
        print(f"- {task}")


def main() -> int:
    args = parse_args()
    if args.list_tasks:
        print_tasks()
        return 0

    config = build_config(args)
    ensure_engine_ready(config)
    env, cwd = build_env(config)

    print_summary(config)

    selected_load_run = config["load_run"]
    selected_checkpoint_file = config["checkpoint_file"]

    if config["engine"] == "host":
        if config["mode"] in ("train", "both"):
            run_command(build_host_train_command(config), env, cwd, dry_run=args.dry_run)
            if config["mode"] == "both" and not selected_load_run:
                selected_load_run = find_latest_host_run(config["experiment_name"])
                print(f"\n[INFO] Ultimo run trovato per il render: {selected_load_run}")

        if config["mode"] == "play" and not selected_load_run:
            selected_load_run = find_latest_host_run(config["experiment_name"])
            print(f"\n[INFO] Ultimo run trovato per il render: {selected_load_run}")

        if config["mode"] in ("play", "both"):
            run_command(
                build_host_play_command(config, selected_load_run),
                env,
                cwd,
                dry_run=args.dry_run,
            )
        return 0

    if config["mode"] in ("train", "both"):
        run_command(build_mjlab_train_command(config), env, cwd, dry_run=args.dry_run)
        if config["mode"] == "both" and not selected_checkpoint_file:
            selected_checkpoint_file = find_latest_mjlab_checkpoint(
                config["experiment_name"],
                selected_load_run,
            )
            print(f"\n[INFO] Ultimo checkpoint trovato per il render: {selected_checkpoint_file}")

    if config["mode"] == "play" and not selected_checkpoint_file:
        selected_checkpoint_file = find_latest_mjlab_checkpoint(
            config["experiment_name"],
            selected_load_run,
        )
        print(f"\n[INFO] Ultimo checkpoint trovato per il render: {selected_checkpoint_file}")

    if config["mode"] in ("play", "both"):
        if not selected_checkpoint_file:
            raise RuntimeError("Impossibile determinare il checkpoint da usare per il play.")
        run_command(
            build_mjlab_play_command(config, selected_checkpoint_file),
            env,
            cwd,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
