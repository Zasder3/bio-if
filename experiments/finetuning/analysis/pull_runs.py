import wandb
import pandas as pd
from tqdm import tqdm


def main():
    # Initialize wandb API
    api = wandb.Api()

    # Replace with your project name
    project_name = "bio-if"

    # Get all runs for the project
    runs = api.runs(project_name)

    def filter_fn(run):
        # Filter out runs that are not finetuning runs
        return run.state == "finished" and run.name.endswith("full")

    runs = list(filter(filter_fn, runs))
    stats_by_dms = {}
    for run in tqdm(runs, desc="Processing runs"):
        config = run.config
        dms = config["study"]
        if dms not in stats_by_dms:
            stats_by_dms[dms] = {}
        for row in run.scan_history():
            if row.get("zero_shot_fitness_spearman") is not None:
                stats_by_dms[dms][row["epoch"]] = row

    # Print out the average zero_shot_fitness_spearman by epoch
    avg_by_epoch = {}
    for _, stats in stats_by_dms.items():
        for epoch, row in stats.items():
            if epoch not in avg_by_epoch:
                avg_by_epoch[epoch] = []
            avg_by_epoch[epoch].append(row["zero_shot_fitness_spearman"])

    for epoch, values in avg_by_epoch.items():
        print(epoch, sum(values) / len(values))

    # turn this into a csv with the following columns:
    # dms, epoch, avg_zero_shot_fitness_spearman
    # dms = "study" in config
    # epoch = epoch
    # avg_zero_shot_fitness_spearman = sum(values) / len

    # Prepare data for DataFrame
    data = []
    for dms, stats in stats_by_dms.items():
        for epoch, row in stats.items():
            data.append(
                {
                    "dms": dms,
                    "epoch": epoch,
                    "avg_zero_shot_fitness_spearman": row["zero_shot_fitness_spearman"],
                }
            )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by DMS and epoch
    df = df.sort_values(["dms", "epoch"])

    # Write to CSV
    csv_filename = "zero_shot_fitness_spearman_by_dms_epoch.csv"
    df.to_csv(csv_filename, index=False)

    print(f"CSV file '{csv_filename}' has been created using pandas.")


if __name__ == "__main__":
    main()
