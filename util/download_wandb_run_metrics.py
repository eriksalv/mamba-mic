import pandas as pd
import wandb

api = wandb.Api()

runs = api.runs("eriksalv-ntnu/camus", filters={"group": "patch4", "jobType": "train"})
ckpt_key = "val_loss"
other_keys = [
    "val_dice_es",
    "val_dice_ed",
    "val_hd95_ed",
    "val_hd95_es",
    "epoch",
    "val_loss",
]
mode = "min"
output_path = "results/camus_patch4.csv"

run_list = []
for run in runs:
    history = run.scan_history(keys=other_keys + [ckpt_key])
    ckpt_vals = [row[ckpt_key] for row in history]
    if mode == "min":
        ckpt_idx = ckpt_vals.index(min(ckpt_vals))
    elif mode == "max":
        ckpt_idx = ckpt_vals.index(max(ckpt_vals))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    metrics = {}
    for key in other_keys:
        metrics[key] = [row[key] for row in history][ckpt_idx]

    run_list.append({"name": run.name, ckpt_key: ckpt_vals[ckpt_idx], **metrics})

runs_df = pd.DataFrame(run_list)
runs_df.to_csv(output_path)
