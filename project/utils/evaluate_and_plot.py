from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from project.utils import calibration

import seaborn as sns
import numpy as np


def brier_score(confidences, true_labels):
    """
    This functions is based on the function of the Keras uncertainty library by Matias Valdenegro-toro:
    https://github.com/mvaldenegro/keras-uncertainty/blob/4269ad3a043485273582bdf23b7ad8d82b41e216/keras_uncertainty/losses.py

    Calculate the Brier score for multi-class classification.

    Parameters:
    - predictions: a 2D numpy array where each row contains predicted probabilities for each class.
    - true_labels: a 1D numpy array where each element is the integer label of the true class (0 to 3 for four classes).

    Returns:
    - The Brier score for the provided predictions and true labels.
    """

    # Convert true_labels to a numpy array if it is not already
    true_labels = np.array(true_labels)

    # Encode categorical labels to integers
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)

    # confidences will be a 2d array in which every array contains the confidences for all possible classes
    if confidences.ndim != 2:
        raise ValueError("Predictions array must be two-dimensional, got: ", confidences.ndim)

    if len(confidences) != len(true_labels_encoded):
        raise ValueError("The length of predictions must match the length of true labels")

    true_probabilities = np.zeros_like(confidences)

    # Assign 1 to the index of the true class
    true_probabilities[np.arange(len(true_labels_encoded)), true_labels_encoded] = 1

    return np.mean(np.square(confidences - true_probabilities))


def plot_confusion_and_evaluate(y_pred, y_true, subject_id, dataset_id, save=True):
    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix subject {subject_id}, dataset {dataset_id}")
    if save:
        plt.savefig(f"./graphs/confusion_plots/dataset{dataset_id}/confusion_subject{subject_id}.png")
        f = open(f"./results/dataset{dataset_id}/evaluation_subject{subject_id}.txt", "w")
        f.write(f"Subject {subject_id} Validation accuracy: {accuracy}\n")
        f.write(f'F1 score subject{subject_id}: {f1}\n')
        f.close()

    else:
        plt.show()
        print(f"Subject {subject_id} Validation accuracy: {accuracy}\n")
        print(f'F1 score subject{subject_id}: {f1}\n')

    plt.clf()

    return


def evaluate_uncertainty(y_predictions, y_test, confidences, subject_id, dataset_id, save=True):
    prediction_confidences = np.max(confidences, axis=1)
    overall_confidence = np.mean(prediction_confidences)

    brier = brier_score(confidences, y_test)

    ece = calibration.get_ece(y_predictions, y_test, prediction_confidences)
    mce = calibration.get_mce(y_predictions, y_test, prediction_confidences)
    nce = calibration.get_nce(y_predictions, y_test, prediction_confidences)
    print(ece)

    if save:
        f = open(f"./results/dataset{dataset_id}/evaluation_subject{subject_id}.txt", "a")
        f.write(f"Overall Confidence {subject_id}: {overall_confidence}\n")
        f.write(f'Brier score subject{subject_id}: {brier}\n')
        f.write(f"ECE {subject_id}: {ece}\n")
        f.write(f"MCE {subject_id}: {mce}\n")
        f.write(f"NCE {subject_id}: {nce}\n")
    else:
        print(f"Overall Confidence {subject_id}: {overall_confidence}\n")
        print(f'Brier score subject{subject_id}: {brier}\n')
        print(f"ECE {subject_id}: {ece}\n")
        print(f"MCE {subject_id}: {mce}\n")
        print(f"NCE {subject_id}: {nce}\n")

def plot_calibration(y_predictions, y_test, confidences, subject_id, dataset_id, save=True):
    prediction_confidences = np.max(confidences, axis=1)
    calibration.plot_calibration_curve(y_predictions, y_test, prediction_confidences,
                                       subject_id=subject_id, dataset_id=dataset_id, save=save)
    plt.clf()
    return

