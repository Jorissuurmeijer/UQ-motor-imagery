import pickle as pkl
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001, Zhou2016, BNCI2014_004, BNCI2014_002
from pyriemann.estimation import Covariances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
import time


from project.utils import calibration
from project.utils.calibration import plot_calibration_curve
from project.utils.evaluate_and_plot import evaluate_uncertainty, plot_confusion_and_evaluate, plot_calibration, \
    brier_score
from project.utils.load_data import load_data
from project.utils.uncertainty_utils import find_best_temperature

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    temperature_scaling = False
    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()
    dataset4 = BNCI2014_001()

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset nor the dict of get_data can get the number of classes

    all_predictions = []
    all_test_labels = []
    train_times = []
    inference_times = []
    inference_counts = []
    for dataset, num_class in zip(datasets, n_classes):
        num_subjects = len(dataset.subject_list)
        all_predictions.append([])
        all_test_labels.append([])
        for subject_id in range(1, num_subjects + 1):
            dataset_id = datasets.index(dataset) + 1

            X, y, metadata = load_data(dataset, subject_id, num_class)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = make_pipeline(CSP(n_components=8), LDA())

            start_time = time.time()
            model.fit(X_train, y_train)
            train_times.append(time.time() - start_time)

            # Predict the labels for the test set
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_times.append(time.time() - start_time)
            inference_counts.append(y_pred.shape[0])

            # Determine the confidence of the model
            if temperature_scaling:
                y_pred_train = model.predict(X_train)
                distances_train = -model.transform(X_train) ** 2
                temperature = find_best_temperature(y_pred_train, y_train, distances_train)

                distance_pred = model.transform(X_test)
                distances = -distance_pred ** 2
                prediction_proba = softmax(distances / temperature)
            else:
                prediction_proba = model.predict_proba(X_test)

            all_predictions[dataset_id - 1].append(prediction_proba)
            all_test_labels[dataset_id - 1].append(y_test)

            # plot and evaluate
            plot_confusion_and_evaluate(y_pred, y_test,
                                        subject_id=subject_id, dataset_id=dataset_id, save=True)

            evaluate_uncertainty(y_pred, y_test, prediction_proba,
                                 subject_id=subject_id, dataset_id=dataset_id, save=True)

            plot_calibration(y_pred, y_test, prediction_proba,
                             subject_id=subject_id, dataset_id=dataset_id, save=True)

    plt.style.use('classic')
    results = {
        "Brier": [],
        "Brier_std": [],
        "ECE": [],
        "ECE_std": [],
        "NCE": [],
        "NCE_std": [],
        "Accuracy": [],
        "Accuracy_std": [],
    }

    print(f"Average train time: {np.mean(train_times)}")
    print(f"Average inference time: {np.mean(np.array(inference_times) / np.array(inference_counts))} per sample")

    calibration_curves = []
    for dataset_id, (dataset_predictions, dataset_labels) in enumerate(zip(all_predictions, all_test_labels)):
        brier_scores = []
        eces = []
        nces = []
        accuracies = []

        all_preds = []
        all_labels = []
        all_confidences = []
        for subject_predictions, subject_labels in zip(dataset_predictions, dataset_labels):
            prediction_confidences = np.max(subject_predictions, axis=1)
            preds = subject_predictions.argmax(axis=1)

            # # We need to do some trickery to make sure we also include the non-predicted values
            # # The "predicted" label that corresponds with each class
            # all_preds.extend(np.repeat(np.arange(0, n_classes[dataset_id], 1)[None,:], len(preds), axis=0).flatten())
            # # The ground truth, repeated for each class
            # all_labels.extend(np.repeat(subject_labels, n_classes[dataset_id]))
            # # All confidences (not only the max)
            # all_confidences.extend(subject_predictions)

            all_preds.extend(preds)
            all_labels.extend(subject_labels)
            all_confidences.extend(prediction_confidences)

            brier_scores.append(brier_score(subject_predictions, subject_labels))
            eces.append(calibration.get_ece(preds, subject_labels, prediction_confidences))
            nces.append(calibration.get_nce(preds, subject_labels, prediction_confidences))
            accuracies.append(accuracy_score(subject_labels, preds))

        results["Brier"].append(np.mean(brier_scores))
        results["Brier_std"].append(np.std(brier_scores))
        results["ECE"].append(np.mean(eces))
        results["ECE_std"].append(np.std(eces))
        results["NCE"].append(np.mean(nces))
        results["NCE_std"].append(np.std(nces))
        results["Accuracy"].append(np.mean(accuracies))
        results["Accuracy_std"].append(np.std(accuracies))

        calibration_curves.append(plot_calibration_curve(np.array(all_preds),
                                                         np.array(all_labels),
                                                         np.array(all_confidences),
                                                         subject_id="", dataset_id=dataset_id+1, save=True))

    results = pd.DataFrame(results)
    if temperature_scaling:
        results.to_csv("./results/CSPLDA-T_results.csv", index=False)
    else:
        results.to_csv("./results/CSPLDA_results.csv", index=False)
    print(results)
    print(results.mean())

    matplotlib.rc_file_defaults()

    pkl.dump(calibration_curves, open("./results/CSPLDA-calibration_curves.pkl", "wb"))

    plt.plot([0, 1], [0, 1], color='black', alpha=0.5, linestyle='dashed', label='_nolegend_')
    for x, y in calibration_curves:
        plt.plot(x, y, alpha=1, linewidth=3)
    plt.xlabel("Confidence", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(["Steryl", "Zhou", "BCIC4-2b", "BCIC4-2a"], fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if temperature_scaling:
        plt.savefig(f"./graphs/calibration_plots/CSPLDA-T.pdf", bbox_inches='tight')
        pkl.dump(calibration_curves, open("./results/CSPLDA-T-calibration_curves.pkl", "wb"))

    else:
        plt.savefig(f"./graphs/calibration_plots/CSPLDA.pdf",bbox_inches='tight')
        pkl.dump(calibration_curves, open("./results/CSPLDA-calibration_curves.pkl", "wb"))

    plt.clf()


if __name__ == '__main__':
    main()
