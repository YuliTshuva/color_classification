"""
Yuli Tshuva.
Overnight run of train08 experiments.
Runs one job per GPU; when a GPU frees up, the next job starts.

Current queue:
  1. seed_eval on pregnancy - the best config of each of the 6 variants
     retrained over 5 seeds (error bars for the exp08 table).
  2. The 6-variant comparison on the 8 disease datasets (60 trials each) -
     does the same-patient relation win generalize beyond pregnancy?

Usage (after connecting, e.g.
  srun -p killable --gres=gpu:a5000:2 -c 50 --mem=16G -t 24:00:00 --pty bash
and activating the venv):
  python run_all.py

Progress: tail -f logs/<job name>.log
Interrupted study jobs are cheap to rerun: already-tried hp configs are
skipped via results/exp08/diseases/<variant>/hp_results.json.
"""

import os
import subprocess
import sys
import time

TRIALS = 60
# (job name, command) pairs, launched in order
JOBS = [("pregnancy_seed_eval",
         [sys.executable, "seed_eval.py", "--dataset", "pregnancy", "--seeds", "5"])]
JOBS += [(f"diseases_{v}",
          [sys.executable, "train08.py", "--variant", v, "--dataset", "diseases",
           "--trials", str(TRIALS)])
         for v in ["rgat", "rgcn", "feat", "feat_color", "pool", "attention"]]


def n_gpus():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def main():
    os.makedirs("logs", exist_ok=True)
    gpus = n_gpus()
    print(f"Found {gpus} GPU(s). Running {len(JOBS)} jobs, one per GPU at a time.")

    queue = list(JOBS)
    running = {}  # gpu -> (process, job_name, log_file)
    failed = []

    while queue or running:
        # Launch on every free GPU
        for gpu in range(gpus):
            if gpu not in running and queue:
                name, cmd = queue.pop(0)
                log = open(os.path.join("logs", f"{name}.log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[gpu] = (proc, name, log)
                print(f"Launched {name} on GPU {gpu} (pid {proc.pid})", flush=True)

        # Collect finished processes
        for gpu in list(running):
            proc, name, log = running[gpu]
            if proc.poll() is not None:
                log.close()
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"{name}: {status}", flush=True)
                if proc.returncode != 0:
                    failed.append(name)
                del running[gpu]

        time.sleep(10)

    if failed:
        print(f"Failed jobs: {failed} - check their logs and rerun.")
    else:
        print("All done. Export the results/exp08/ folder.")


if __name__ == "__main__":
    main()
