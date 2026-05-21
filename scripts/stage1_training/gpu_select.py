import argparse
import csv
import subprocess


def query_gpus() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 4:
            continue
        index, mem_used, mem_total, util = [item.strip() for item in row]
        rows.append(
            {
                "index": int(index),
                "memory_used": int(mem_used),
                "memory_total": int(mem_total),
                "utilization": int(util),
            }
        )
    return rows


def pick_gpu(max_memory_used_mb: int | None = None) -> int:
    gpus = query_gpus()
    if max_memory_used_mb is not None:
        filtered = [gpu for gpu in gpus if gpu["memory_used"] <= max_memory_used_mb]
        if filtered:
            gpus = filtered
    if not gpus:
        raise RuntimeError("No visible NVIDIA GPUs found.")
    return min(gpus, key=lambda gpu: (gpu["memory_used"], gpu["utilization"]))["index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_memory_used_mb", type=int, default=None)
    args = parser.parse_args()
    print(pick_gpu(args.max_memory_used_mb))


if __name__ == "__main__":
    main()
