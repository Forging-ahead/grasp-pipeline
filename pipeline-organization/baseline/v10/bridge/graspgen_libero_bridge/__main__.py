import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Independent GraspGen/cuRobo LIBERO bridge")
    parser.add_argument("command", choices=["doctor", "run"])
    parser.add_argument("--config", type=Path, default=root / "configs/server.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-mode", choices=["geometry", "rgbd"])
    parser.add_argument("--init-state", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate-offset", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key, value in [("input_mode", args.input_mode), ("init_state_id", args.init_state),
                       ("seed", args.seed), ("candidate_offset", args.candidate_offset)]:
        if value is not None:
            config[key] = value
    gpu = str(config.get("simulation_gpu", 6))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUTF8"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(config.get("cpu_threads", 4))
    output = args.output or root / "runs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    from .evaluate import execute
    result = execute(config, output, doctor=args.command == "doctor")
    print(json.dumps({"output": str(output.resolve()), "status": result["status"], "task_success": result["task_success"]}), flush=True)
    return 0 if result["status"] == "doctor_passed" or result.get("pipeline_success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
