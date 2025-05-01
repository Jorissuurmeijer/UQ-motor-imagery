import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from cycler import cycler

def use_classic_with_tab10() -> None:
    # Classic axes but with new color codes
    plt.style.use("classic")            # fonts, spines, ticks, ...
    tab10 = plt.rcParamsDefault['axes.prop_cycle'].by_key()['color']
    plt.rcParams['axes.prop_cycle'] = cycler(color=tab10)


def plot_per_dataset(npz_file: str ,
                     out_dir: str = "./plots/by_dataset",
                     line_width: float = 3.0) -> None:
    curves = np.load(npz_file, allow_pickle=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_keys = {}
    for key in curves.files:
        ds_idx = key.split("_")[0]          # "ds0" → "ds0"
        ds_keys.setdefault(ds_idx, []).append(key)

    for ds_idx, keys in ds_keys.items():
        use_classic_with_tab10()
        plt.figure(figsize=(5, 4))

        for k in keys:
            cov, acc = curves[k]
            rej = 1.0 - cov
            label = k.split("_", 1)[1].replace("_", " ")
            plt.plot(rej, acc, lw=line_width, label=label)

        plt.xlim(0, 1.05); plt.ylim(0, 1.05)
        plt.xlabel("Rejection rate")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left", fontsize=11)
        plt.tight_layout()

        fname = out_dir / f"{ds_idx}_accuracy_coverage.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"saved {fname}")


def plot_average_over_datasets(npz_file: str ,
                               out_file: str = "./plots/all_datasets_average.pdf",
                               line_width: float = 3.0,
                               grid_step: float = 0.01) -> None:

    curves = np.load(npz_file, allow_pickle=True)

    # --- collect per-method curves ----------------------------------------
    method_curves = {}
    for key in curves:
        cov, acc = curves[key]
        rej = 1.0 - cov
        method = key.split("_", 1)[1]           # part after "dsX_"
        method_curves.setdefault(method, []).append((rej, acc))

    rej_grid = np.arange(0.0, 1.0 + 1e-6, grid_step)   # 0 to 1 in equal steps, in this case 1 percent per step
    use_classic_with_tab10()
    plt.figure(figsize=(5, 4))

    for method, lst in method_curves.items():
        acc_interp = []
        for rej, acc in lst:
            f = interp1d(rej, acc, kind="previous", bounds_error=False,
                         fill_value=(acc[0], acc[-1]))
            acc_interp.append(f(rej_grid))
        acc_interp = np.vstack(acc_interp)      # (n_ds, n_grid)

        mean_acc = acc_interp.mean(axis=0)
        plt.plot(rej_grid, mean_acc,
                 lw=line_width,
                 label=method.replace("_", " "))

    plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.xlabel("Rejection rate")
    plt.ylabel("Mean accuracy across datasets")
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"saved {out_file}")


def main():
    plot_per_dataset("graphs/accuracy_coverage/all_accuracy_coverage_curves.npz")

    plot_average_over_datasets(
        "graphs/accuracy_coverage/all_accuracy_coverage_curves.npz",
        out_file="plots/average_over_datasets.pdf"
    )

if __name__ == "__main__":
    main()