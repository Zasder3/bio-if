import queue
import pandas as pd
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


def run_main(dms_id, gpu_id, batch_size):
    if batch_size < 1:
        return
    print(f"Running {dms_id} on GPU {gpu_id} with batch size {batch_size}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = [
        "python",
        "src/bio_if/training/main.py",
        "--dms_id",
        dms_id,
        "--train_data_path",
        "experiments/finetuning/datasets/dms_ids_search.parquet.gzip",
        "--ood_val_data_path",
        "experiments/finetuning/datasets/uniref50_random_10k.csv",
        "--fitness_dataset_path",
        f"experiments/dms/studies/{dms_id}.csv",
        "--fp16",
        "--lr",
        "1e-6",
        "--batch_size",
        str(batch_size),
    ]
    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError:
        print(f"OOM error for {dms_id}. Retrying with half batch size.")
        run_main(dms_id, gpu_id, batch_size // 2)


def gpu_task(dms_id, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        run_main(dms_id, gpu_id, 32)
    finally:
        gpu_queue.put(gpu_id)


def main():
    df = pd.read_csv("experiments/dms/DMS_substitutions.csv")
    dms_ids = df["DMS_id"].unique().tolist()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    gpu_queue = queue.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for dms_id in dms_ids:
            future = executor.submit(gpu_task, dms_id, gpu_queue)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with error: {e}")


if __name__ == "__main__":
    main()
