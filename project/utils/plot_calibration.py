import pickle as pkl

from matplotlib import pyplot as plt


def plot_calibration_from_file(filepath: str, model_name: str):
    calibration_curves = pkl.load(open(filepath, 'rb'))
    plt.plot([0, 1], [0, 1], color='black', alpha=0.5, linestyle='dashed', label='_nolegend_')
    for x, y in calibration_curves:
        plt.plot(x, y, alpha=0.8, linewidth=3)
    plt.xlabel("Confidence", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.legend(["Steryl", "Zhou", "BCIC4-2b", "BCIC4-2a"], fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(f"./graphs/calibration_plots/{model_name}.pdf", bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    for model_name in ["CSPLDA", "MDRM", "MDRM-T", "DUQ", "DE"]:
        filepath = f'./results/{model_name}-calibration_curves.pkl'
        plot_calibration_from_file(filepath, model_name)