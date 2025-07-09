import pickle as pkl
import warnings

import matplotlib
import numpy as np
import pandas as pd
import time
from keras_uncertainty.utils import entropy
from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001, BNCI2014_002, Zhou2016, BNCI2014_004
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

from project.models.shallowConvNet.DUQ.SCN_model_DUQ import ShallowConvNet
from project.utils import calibration
from project.utils.calibration import plot_calibration_curve
from project.utils.evaluate_and_plot import plot_confusion_and_evaluate, evaluate_uncertainty, plot_calibration, \
    brier_score
from project.utils.load_data import load_data
from project.utils.uncertainty_utils import find_best_temperature

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()
    dataset4 = BNCI2014_001()  # original one

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset nor the dict of get_data can get the number of classes

    channels = [15, 14, 3, 22]        # the same holds here
    samples_data = [2561, 1251, 1126, 1001]

    all_predictions = []
    all_test_labels = []
    train_times = []
    inference_times = []
    inference_counts = []
    for dataset, num_class, chans, samples in zip(datasets, n_classes, channels, samples_data):
        num_subjects = len(dataset.subject_list)
        all_predictions.append([])
        all_test_labels.append([])

        for subject_id in tqdm(range(1, num_subjects + 1)):       # loop to take data and train model per subject
            dataset_id = datasets.index(dataset) + 1

            X, y, _ = load_data(dataset, subject_id, num_class)
            assert not np.isnan(X).any(), "Data contains NaN values"
           
            unique_labels = np.unique(y)
            num_unique_labels = len(unique_labels)
            assert num_unique_labels == num_class, "The number of labels does not match the expected number of classes."

            X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

            label_encoder = LabelEncoder()
            y_integers = label_encoder.fit_transform(y)
            y_categorical = utils.to_categorical(y_integers, num_classes=num_unique_labels)

            X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
            assert not np.isnan(X_train).any(), "Training data contains NaN values"
            assert not np.isnan(X_test).any(), "Test data contains NaN values"

            net = ShallowConvNet()
            model = net.build(nb_classes=num_class, Chans=chans, Samples=samples, dropoutRate=0.5)

            start_time = time.time()
            # weights = compute_sample_weight('balanced', y=y_train)  # can be used when wanting to use balanced weights
            model.fit(
                X_train,
                y_train,
                callbacks=[early_stopping],
                epochs=200,  # Should be 200 for proper run
                batch_size=64, validation_split=0.1,   # sample_weight=weights,
                verbose=0,
            )
            train_times.append(time.time() - start_time)

            model.save(f'./saved_trained_models/SCN/PerSubject/subject{subject_id}')  # use to save the model

            start_time = time.time()
            predictions_test = model.predict(X_test)
            prediction_proba = predictions_test
            # assert not np.isnan(predictions_test).any(), "Model predictions contain NaN values"
            #
            inference_times.append(time.time() - start_time)
            inference_counts.append(predictions_test.shape[0])
            #
            predicted_classes = np.argmax(predictions_test, axis=1)
            # assert not np.isnan(predicted_classes).any(), "Model predicted classes contain NaN values"
            #
            # # Calculate probabilities with a softmax using a temperature to determine the confidence of the model
            # distances_train = model.predict(X_train) ** 2
            # temperature = find_best_temperature(distances_train.argmax(axis=1), y_train.argmax(axis=1), distances_train)
            #
            # distances_test = predictions_test ** 2
            # prediction_proba = softmax(distances_test / temperature)

            entr = entropy(y_test, prediction_proba)        # not used for now
            print("Entropy: ", entr)
            y_test = y_test.argmax(axis=1)

            all_predictions[dataset_id - 1].append(prediction_proba)
            all_test_labels[dataset_id - 1].append(y_test)

            # plot and evaluate
            plot_confusion_and_evaluate(predicted_classes, y_test,
                                        subject_id=subject_id, dataset_id=dataset_id, save=True)

            evaluate_uncertainty(predicted_classes, y_test, prediction_proba,
                                 subject_id=subject_id, dataset_id=dataset_id, save=True)

            plot_calibration(predicted_classes, y_test, prediction_proba,
                             subject_id=subject_id, dataset_id=dataset_id, save=True)

    print(f"Average train time: {np.mean(train_times)}")
    print(f"Average inference time: {np.mean(np.array(inference_times) / np.array(inference_counts))} per sample")

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

            # We need to do some trickery to make sure we also include the non-predicted values
            # The "predicted" label that corresponds with each class
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
    results.to_csv("./results/DUQ-no-softmax_results.csv", index=False)

    print(results)
    print(results.mean())

    matplotlib.rc_file_defaults()

    pkl.dump(calibration_curves, open("./results/DUQ-no-softmax-calibration_curves.pkl", "wb"))

    plt.plot([0, 1], [0, 1], color='black', alpha=0.5, linestyle='dashed', label='_nolegend_')
    for x, y in calibration_curves:
        plt.plot(x, y, alpha=0.8, linewidth=3)
    plt.xlabel("Confidence", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend(["Steryl", "Zhou", "BCIC4-2b", "BCIC4-2a"], fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(f"./graphs/calibration_plots/DUQ-no-softmax.pdf", bbox_inches='tight')
    pkl.dump(calibration_curves, open("./results/DUQ-no-softmax-calibration_curves.pkl", "wb"))

    plt.clf()


if __name__ == '__main__':
    main()
