import matplotlib
import pandas as pd
from keras import backend as K
from keras import Model, utils, activations, optimizers
from keras import callbacks



#
# from keras._tf_keras.keras.models import Model
# from keras._tf_keras.keras import utils
# from keras._tf_keras.keras.callbacks import EarlyStopping
# from sklearn.utils.extmath import softmax
# # from keras.optimizers import Adam

from matplotlib import pyplot as plt
from moabb.datasets import BNCI2014_001, BNCI2014_002, Zhou2016, BNCI2014_004
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_uncertainty.utils import entropy

import pickle as pkl

# from keras._tf_keras.keras.utils import to_categorical

from project.utils import calibration
from project.utils.calibration import plot_calibration_curve
from project.utils.load_data import load_data
from project.models.shallowConvNet.standard.standard_SCN_model import ShallowConvNet
from project.utils.evaluate_and_plot import plot_confusion_and_evaluate, evaluate_uncertainty, plot_calibration, \
    brier_score

from tqdm import tqdm
import numpy as np

import warnings

from project.utils.uncertainty_utils import find_best_temperature

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    temperature_scaling = False #TODO fout eruit halen

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()       # the bad performing one, if done again, take in range, 3,4
    dataset4 = BNCI2014_001()  # original one

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset nor the dict of get_data can get the number of classes

    channels = [15, 14, 3, 22]  # the same holds here
    samples_data = [2561, 1251, 1126, 1001]

    num_models = 1

    all_predictions = []
    all_test_labels = []
    for dataset, num_class, chans, samples in zip(datasets, n_classes, channels, samples_data):
        num_subjects = len(dataset.subject_list)
        all_predictions.append([])
        all_test_labels.append([])

        for subject_id in range(1, num_subjects + 1):
            dataset_id = datasets.index(dataset) + 1

            X, y, metadata = load_data(dataset, subject_id, num_class)

            unique_labels = np.unique(y)
            num_unique_labels = len(unique_labels)
            assert num_unique_labels == num_class, "The number of labels does not match the expected number of classes."

            X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = utils.to_categorical(y, num_classes=num_unique_labels)

            X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

            predictions = np.zeros((num_models, X_test.shape[0], num_class))
            train_predictions = np.zeros((num_models, X_train.shape[0], num_class))
            for model_idx in tqdm(range(num_models)):

                model = ShallowConvNet(nb_classes=num_class, Chans=chans, Samples=samples, dropoutRate=0.5)
                optimizer = optimizers.Adam(learning_rate=0.001)  # standard 0.001
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                # weights = compute_sample_weight('balanced', y=y_train)    # can be used for balanced weights
                model.fit(
                    X_train,
                    y_train,
                    callbacks=[early_stopping],
                    epochs=100,  # Should be 100
                    batch_size=64, validation_split=0.1,   # sample_weighTt=weights,
                    verbose=0,
                )

                if temperature_scaling:
                    assert model.layers[0].input is not None and len(model.layers[0].input) is not 0
                    logits_layer_model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
                    predictions[model_idx] = logits_layer_model.predict(X_test).squeeze()
                    train_predictions[model_idx] = logits_layer_model.predict(X_train).squeeze()
                else:
                    predictions[model_idx] = model.predict(X_test)

            mean_predictions = np.mean(np.array([predictions]), axis=0).squeeze()

            if temperature_scaling:
                mean_train_logits = np.mean(np.array([train_predictions]), axis=0).squeeze()

                y_class_train = mean_train_logits.argmax(axis=1)
                temperature = find_best_temperature(y_class_train, y_train.argmax(axis=1), mean_train_logits)

                prediction_proba = activations.softmax(mean_predictions / temperature)
            else:
                prediction_proba = mean_predictions
            predicted_classes = np.argmax(mean_predictions, axis=1)
            # confidence = np.max(max_pred_0, axis=1)

            # entr = entropy(y_test, prediction_proba)       # not further used for now
            # print("Entropy: ", entr)
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
    results.to_csv("./results/CNN-T_results.csv", index=False)

    print(results)
    print(results.mean())

    matplotlib.rc_file_defaults()

    pkl.dump(calibration_curves, open("./results/CNN-T-calibration_curves.pkl", "wb"))

    plt.plot([0, 1], [0, 1], color='black', alpha=0.5, linestyle='dashed', label='_nolegend_')
    for x, y in calibration_curves:
        plt.plot(x, y, alpha=0.8, linewidth=3)
    plt.xlabel("Confidence", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(["Steryl", "Zhou", "BCIC4-2b", "BCIC4-2a"], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(f"./graphs/calibration_plots/CNN-T.pdf")
    pkl.dump(calibration_curves, open("./results/CNN-T-calibration_curves.pkl", "wb"))

    plt.clf()


if __name__ == '__main__':
    main()
