import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.python.keras import backend as K

from src.utils import get_art_peaks, align_lists, bland_altman_plot


class OutputPredictionCallback(Callback):
    """"
    Callback to write predictions of model to file
    """

    def __init__(self, train_generator, val_generator, num_batches=1, save_dir=None):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.num_batches = num_batches  # number of batches to predict
        if save_dir is None:
            self.save_dir = "plots_" + datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_waveforms(self, epoch, train_or_val="train", num_images=10, padding_size=4):
        """
        Plot input waveforms and true/predicted waveforms

        :param epoch: training epoch number
        :param train_or_val: use "train" generator or "validation" generator
        :param num_images: number of images to generate
        :param padding_size: padding size of the window
        :param save_dir: path to save files
        :return: None
        """
        if train_or_val == "train":
            print("=" * 40)
            print("TRAINING SET")
            print("=" * 40)
            generator = self.train_generator
            fig_file_string = "epoch_{}_pred_img_train_{}_{}.png"
        elif train_or_val == "val":
            print("=" * 40)
            print("VALIDATION SET")
            print("=" * 40)
            generator = self.val_generator
            fig_file_string = "epoch_{}_pred_img_val_{}_{}.png"
        else:
            print("ERROR: train_or_val should be set to either 'train' or 'val'")
            return None

        image_count = 0
        num_batches = max(int(num_images / generator.batch_size), 1)
        index = np.random.randint(low=0, high=10000, size=num_batches)
        # for each batch
        for i in index:
            # get a batch from the generator
            X, y_true = generator.__getitem__(i)
            # y_true contains both the true ABP waveform and median sys, dias, mean BP for window
            y_true = y_true
            # for each window in the batch
            for j in range(X.shape[0]):
                if image_count == num_images:
                    return
                image_count += 1
                print("y true shape:", y_true.shape)
                if len(y_true.shape) < 3:
                    test_df = pd.DataFrame(X[j])
                    print(X.shape)
                    print(y_true.shape)
                    print(test_df.shape)
                    test_df["true_sys"], test_df["true_dias"], test_df["true_mean"] = generator.y_scaler.inverse_transform(y_true[j])
                    test_df["pred_sys"], test_df["pred_dias"], test_df["pred_mean"] = generator.y_scaler.inverse_transform(
                        self.model.predict_on_batch(X)[j])
                    test_df["y_peaks"] = 0
                else:
                    print(X.shape)
                    # test_df = pd.DataFrame(X[-1, :, 0, :])
                    test_df = pd.DataFrame(X[j, padding_size:y_true[j].shape[0] + padding_size, :])
                    print(test_df.shape)
                    # test_df = pd.DataFrame(y_true, columns=["y_true"])
                    try:
                        test_df["y_peaks"] = y_true[j, :, 1]
                        test_df["y_true"] = generator.y_scaler.inverse_transform(y_true[j, :, 0])
                        test_df["y_pred"] = generator.y_scaler.inverse_transform(
                            self.model.predict_on_batch(X)[j][:, 0])
                    except KeyError as e:
                        print(e)
                        continue

                try:
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
                    # first plot: predicted and true output
                    ax1.plot(test_df.iloc[:, -2], color="g", label="True ABP")
                    ax1.plot(test_df.iloc[:, -1], color="r", label="Predicted ABP", dashes=[8, 2])
                    peaks = np.nonzero(test_df["y_peaks"].values)[0]
                    ax1.legend()
                    ax1.scatter(peaks, test_df["y_true"][peaks], marker='*', s=60, c='black', label='sys/dias BP')
                    bp_max_indices, bp_min_indices = get_art_peaks(test_df["y_pred"])
                    pred_peaks = list(bp_max_indices) + list(bp_min_indices)
                    ax1.scatter(pred_peaks, test_df["y_pred"][pred_peaks], marker='*', s=60, c='black',
                                label='sys/dias BP')
                    # second plot: ECG and wavelet transforms
                    ax2.plot(test_df.iloc[:, 0])
                    ax2.plot(test_df.iloc[:, 3:5], alpha=0.5)
                    ax2.legend(["ECG", "Wavelet - ECG peak", "Wavelet - ECG T wave"])
                    # third plot: PPG and wavelet transforms
                    ax3.plot(test_df.iloc[:, 5])
                    ax3.plot(test_df.iloc[:, 1:3], alpha=0.5)
                    ax3.legend(["PPG", "Wavelet - PPG peak", "Wavelet - PPG d-notch"])
                    nibp_vals = generator.X_scaler.inverse_transform(X[j][:, :6])[:, -3:]
                    print("nibp_vals shape", nibp_vals.shape)
                    plt.annotate("SYS BP: {:.1f} \nDIAS BP: {:.1f} \nMEAN BP: {:.1f}".format(nibp_vals[:, 0].mean(),
                                                                                             nibp_vals[:, 1].mean(),
                                                                                             nibp_vals[:, 2].mean()),
                                 xy=(0.15, 0.80),
                                 xycoords='axes fraction')
                    plt.savefig(os.path.join(self.save_dir, fig_file_string.format(epoch, i, j)))
                    # plt.show()
                    plt.close()
                    # save data used to generate plots
                    save_file = os.path.join(self.save_dir, os.path.splitext(fig_file_string)[0].format(epoch, i, j))+".csv"
                    test_df.to_csv(save_file, header=True, index=False)
                except ValueError as e:
                    print(e)
                    pass
                except KeyError as e:
                    print(e)
                    continue

    def plot_bland_altman(self, epoch, train_or_val="train", wave_or_scaler="wave", num_windows=500):
        """
        Function for generating predictions with the model and comparing against ground truth
        using the Bland-Altman method

        :param epoch: training epoch number
        :param train_or_val: use "train" generator or "validation" generator
        :param wave_or_scaler: choose to plot bland-altman for either "wave" prediction or "scaler" prediction
        :param num_windows: number of windows to get blood pressure values from
        :param padding_size: padding size of the window
        :param save_dir: directory to save plots/data
        :return: None
        """
        # bland-altman y-axis limits
        plot_ylim = [-60, 60]
        plot_xlim_sys = [50, 200]
        plot_xlim_dias = [0, 150]

        y_true_sys_bp_all = []
        y_true_dias_bp_all = []
        y_pred_sys_bp_all = []
        y_pred_dias_bp_all = []

        if train_or_val == "train":
            print("=" * 40)
            print("TRAINING SET")
            print("=" * 40)
            generator = self.train_generator
            sys_title_string = "Bland-Altman - Systolic - Training: {} +/- {}"
            dias_title_string = "Bland-Altman - Diastolic - Training: {} +/- {}"
            fig_file_string = "epoch_{}_bland_altman_train_{}.png"
        elif train_or_val == "val":
            print("=" * 40)
            print("VALIDATION SET")
            print("=" * 40)
            generator = self.val_generator
            sys_title_string = "Bland-Altman - Systolic - Validation: {} +/- {}"
            dias_title_string = "Bland-Altman - Diastolic - Validation: {} +/- {}"
            fig_file_string = "epoch_{}_bland_altman_val_{}.png"
        else:
            print("ERROR: train_or_val should be set to either 'train' or 'val'")
            return None

        window_count = 0
        num_batches = max(int(num_windows / generator.batch_size), 1)
        index = np.random.randint(low=0, high=10000, size=num_batches)
        for i in index:
            X, y_true = generator.__getitem__(i)

            # y_true contains both the true ABP waveform and median sys, dias, mean BP for window
            # get predictions for batch (prediction is tuple: (wave, scaler))
            if wave_or_scaler == "wave":
                y_true = y_true
                y_pred = self.model.predict_on_batch(X)
            elif wave_or_scaler == "scaler":
                y_true = y_true
                y_pred = self.model.predict_on_batch(X)
            # for each window in the batch
            for j in range(X.shape[0]):
                if window_count == num_windows:
                    return
                window_count += 1

                # get values of predicted wave at sys/dias BP indices
                if wave_or_scaler == "wave":
                    # scale BP back to normal units
                    y_true_scaled = generator.y_scaler.inverse_transform(y_true[j, :, 0])

                    # get predictions
                    y_pred_scaled = generator.y_scaler.inverse_transform(y_pred[j])[:, 0]
                    # get indices of sys/dias BP
                    true_bp_max_indices, true_bp_min_indices = get_art_peaks(y_true_scaled)
                    pred_bp_max_indices, pred_bp_min_indices = get_art_peaks(y_pred_scaled)

                    # align bp indices in case of different number of peaks
                    true_bp_max_indices, pred_bp_max_indices = align_lists(true_bp_max_indices, pred_bp_max_indices)
                    true_bp_min_indices, pred_bp_min_indices = align_lists(true_bp_min_indices, pred_bp_min_indices)

                    # get values of blood pressure at peak indices
                    y_true_sys_bp_all = y_true_sys_bp_all + list(y_true_scaled[true_bp_max_indices])
                    y_true_dias_bp_all = y_true_dias_bp_all + list(y_true_scaled[true_bp_min_indices])

                    y_pred_sys_bp_all = y_pred_sys_bp_all + list(y_pred_scaled[pred_bp_max_indices])
                    y_pred_dias_bp_all = y_pred_dias_bp_all + list(y_pred_scaled[pred_bp_min_indices])
                # BP prediction should be a tuple of scalers (systolic, diastolic, mean BP)
                elif wave_or_scaler == "scaler":
                    # scale BP back to normal units
                    y_true_scaled = generator.y_scaler.inverse_transform(y_true[j])
                    y_true_sys_bp_all = y_true_sys_bp_all + list([y_true_scaled[0]])
                    y_true_dias_bp_all = y_true_dias_bp_all + list([y_true_scaled[1]])
                    # get predictions
                    y_pred_scaled = generator.y_scaler.inverse_transform(y_pred[j])
                    y_pred_sys_bp_all = y_pred_sys_bp_all + list([y_pred_scaled[0]])
                    y_pred_dias_bp_all = y_pred_dias_bp_all + list([y_pred_scaled[1]])

        # create two-part figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # left window: systolic BP
        bland_altman_plot(y_true_sys_bp_all, y_pred_sys_bp_all, ax=ax1)
        ax1.legend(["N={}".format(len(y_true_sys_bp_all))], loc='upper left')
        ax1.set_ylim(plot_ylim)
        ax1.set_xlim(plot_xlim_sys)
        diffs = np.array(y_true_sys_bp_all) - np.array(y_pred_sys_bp_all)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, axis=0)
        ax1.title.set_text(sys_title_string.format(np.round(mean_diff, 1), np.round(std_diff, 1)))
        # right window: diastolic BP
        bland_altman_plot(y_true_dias_bp_all, y_pred_dias_bp_all, ax=ax2)
        ax2.legend(["N={}".format(len(y_true_dias_bp_all))], loc='upper left')
        ax2.set_ylim(plot_ylim)
        ax2.set_xlim(plot_xlim_dias)
        diffs = np.array(y_true_dias_bp_all) - np.array(y_pred_dias_bp_all)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, axis=0)
        ax2.title.set_text(dias_title_string.format(np.round(mean_diff, 1), np.round(std_diff, 1)))
        # save figure to file
        plt.savefig(os.path.join(self.save_dir, fig_file_string.format(epoch, wave_or_scaler)))
        # plt.show()
        plt.close()
        # write data used for bland-altman plot to file
        sys_bp_vals = {"sys_true": y_true_sys_bp_all,
                       "sys_pred": y_pred_sys_bp_all}
        save_file_name = os.path.join(self.save_dir, "epoch_{}_{}_sys_bland-altman.csv".format(epoch, wave_or_scaler))
        pd.DataFrame.from_dict(sys_bp_vals, orient="columns").to_csv(save_file_name, header=True, index=False)
        dias_bp_vals = {"dias_true": y_true_dias_bp_all,
                        "dias_pred": y_pred_dias_bp_all}
        save_file_name = os.path.join(self.save_dir, "epoch_{}_{}_dias_bland-altman.csv".format(epoch, wave_or_scaler))
        pd.DataFrame.from_dict(dias_bp_vals, orient="columns").to_csv(save_file_name, header=True, index=False)

    def on_epoch_end(self, epoch, logs={}):
        # randomly select index
        num_images = 20
        num_bland_altman_wave = 2000
        num_bland_altman_scaler = 400

        self.plot_waveforms(epoch, train_or_val="train", num_images=num_images)
        self.plot_waveforms(epoch, train_or_val="val", num_images=num_images)

        self.plot_bland_altman(epoch, train_or_val="train", wave_or_scaler="wave", num_windows=num_bland_altman_wave)
        self.plot_bland_altman(epoch, train_or_val="val", wave_or_scaler="wave", num_windows=num_bland_altman_wave)

        #self.plot_bland_altman(epoch, train_or_val="train", wave_or_scaler="scaler", num_windows=num_bland_altman_scaler)
        #self.plot_bland_altman(epoch, train_or_val="val", wave_or_scaler="scaler", num_windows=num_bland_altman_scaler)


class TrainingPlot(Callback):
    def __init__(self, save_dir=None):
        if save_dir is None:
            self.save_dir = "plots_" + datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('loss.png')
            plt.close()


class GradientCallback(Callback):
    console = False

    file_writer = tf.summary.create_file_writer("./logs")
    file_writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        weights = [w for w in self.model.trainable_weights if 'dense' in w.name and 'bias' in w.name]
        loss = self.model.total_loss
        optimizer = self.model.optimizer
        gradients = optimizer.get_gradients(loss, weights)
        for t in gradients:
            if self.console:
                print('Tensor: {}'.format(t.name))
                print('{}\n'.format(K.get_value(t)[:10]))
            else:
                tf.summary.histogram(t.name, data=t)
