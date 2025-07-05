import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

font = {"size": 20}

matplotlib.rc("font", **font)

if __name__ == "__main__":
    x_log_scale = False

    xs = [32, 64, 128, 256, 900]
    val_loss = {"segresnet": [], "swinunetr": [], "swinumamba": [], "vssd": []}
    val_dice_ed = {"segresnet": [], "swinunetr": [], "swinumamba": [], "vssd": []}
    val_dice_es = {"segresnet": [], "swinunetr": [], "swinumamba": [], "vssd": []}

    df32 = pd.read_csv("results/camus_binary_32.csv")
    df64 = pd.read_csv("results/camus_binary_64.csv")
    df128 = pd.read_csv("results/camus_binary_128.csv")
    df256 = pd.read_csv("results/camus_binary_256.csv")
    df_full = pd.read_csv("results/camus_binary_full.csv")

    for df in [df32, df64, df128, df256, df_full]:
        for model in val_loss.keys():
            val_loss[model].append(df[df["name"].str.startswith(model)].describe().loc["mean", "val_loss"])
            val_dice_ed[model].append(df[df["name"].str.startswith(model)].describe().loc["mean", "val_dice_ed"])
            val_dice_es[model].append(df[df["name"].str.startswith(model)].describe().loc["mean", "val_dice_es"])

    if x_log_scale:
        plt.figure(figsize=(10, 6))
        plt.plot(xs, val_loss["segresnet"], "o-", label="segresnet", linewidth=3)
        plt.plot(xs, val_loss["swinunetr"], "o-", label="swinunetr", linewidth=3)
        plt.plot(xs, val_loss["swinumamba"], "o-", label="swinumamba", linewidth=3)
        plt.plot(xs, val_loss["vssd"], "o-", label="vssd", linewidth=3)
        plt.legend()
        plt.grid()
        plt.xlabel("# Training Samples")
        plt.ylabel("Validation Loss (DiceCE)")
        plt.savefig("val_loss_training_samples.png")
        plt.show()
    else:
        xs = [32, 64, 128, 256, 900]
        plt.figure(figsize=(10, 6))
        plt.plot(np.log2(xs), val_loss["segresnet"], "o-", label="segresnet", linewidth=3)
        plt.plot(np.log2(xs), val_loss["swinunetr"], "o-", label="swinunetr", linewidth=3)
        plt.plot(np.log2(xs), val_loss["swinumamba"], "o-", label="swinumamba", linewidth=3)
        plt.plot(np.log2(xs), val_loss["vssd"], "o-", label="vssd", linewidth=3)
        plt.legend()
        plt.grid()
        plt.xlabel("# Training Samples (log scale)")
        plt.ylabel("Validation Loss (DiceCE)")
        plt.savefig("val_loss_training_samples_log.png")
        plt.show()
