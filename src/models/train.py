#!/usr/bin/env python
import sys
import pickle
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

sys.path.append("../../")
from src.utils import *
from src.generator import DataGenerator
from src.callbacks import OutputPredictionCallback, TrainingPlot, GradientCallback
import src.project_configs as project_configs
import src.models.abp_model as abp_model
import src.utils as utils
import keract
load_scaler_pickle = True
load_weights = False
weight_file = "2020-02-17_12:34:58/weights.01.hdf5"

if __name__ == '__main__':
    # train_test_split_mimic(project_configs.test_dir, project_configs.train_dir, train_split=0.3)

    # create DataGenerator objects
    # optionally load existing scaler objects
    if not load_scaler_pickle:
        train_gen = DataGenerator(data_dir=project_configs.train_dir,
                                  window_len=project_configs.window_size,
                                  batch_size=project_configs.batch_size)
        pickle.dump(train_gen.X_scaler, open("../../models/train_X_scaler.pkl", "wb"))
        pickle.dump(train_gen.y_scaler, open("../../models/train_y_scaler.pkl", "wb"))
    else:
        X_scaler, y_scaler = utils.load_scaler_objects(project_configs.X_scaler_pickle,
                                                       project_configs.y_scaler_pickle)
        train_gen = DataGenerator(data_dir=project_configs.train_dir,
                                  window_len=project_configs.window_size,
                                  batch_size=project_configs.batch_size,
                                  X_scaler=X_scaler,
                                  y_scaler=y_scaler)

    # use mean/stdev from training data to scale testing data
    val_gen = DataGenerator(data_dir=project_configs.val_dir,
                            window_len=project_configs.window_size,
                            batch_size=project_configs.batch_size,
                            X_scaler=train_gen.X_scaler,
                            y_scaler=train_gen.y_scaler)

    assert(not np.any(np.isnan(train_gen.X_scaler.var_)))

    if load_weights:
        save_dir = os.path.dirname(weight_file)
    else:
        save_dir = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
    print("Saving files to {}".format(save_dir))

    model = abp_model.create_model_lstm(save_dir=save_dir)

    # save project configs and abp_model
    log_project_configs(save_dir=save_dir)

    # x = train_gen.__getitem__(0)[0][0]
    # print(x)
    # activations = keract.get_activations(model, np.array([x]))
    #
    # def print_names_and_values(activations: dict):
    #     for layer_name, layer_activations in activations.items():
    #         print(layer_name)
    #         print(layer_activations)
    #         print('')
    #     print('-' * 80)
    # print_names_and_values(activations)
    # # keract.display_activations(activations)
    #
    # exit()
    # instantiate the callback objects
    output_pred = OutputPredictionCallback(train_gen, val_gen, save_dir=save_dir)
    checkpoint = ModelCheckpoint(os.path.join(save_dir, "weights.{epoch:02d}.hdf5"), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=project_configs.batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    # gradients = GradientCallback()
    plot_losses = TrainingPlot(save_dir=save_dir)

    # train the model
    if load_weights:
        model.load_weights(weight_file)
        initial_epoch = int(os.path.basename(weight_file).split(".")[1])
    else:
        initial_epoch = 0
    print("Starting at epoch {}".format(initial_epoch))
    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  validation_steps=50,
                                  steps_per_epoch=500,
                                  epochs=150,
                                  verbose=1,
                                  callbacks=[output_pred, checkpoint, earlystop, tensorboard, plot_losses],
                                  initial_epoch=initial_epoch,
                                  use_multiprocessing=False,
                                  max_queue_size=1000,
                                  workers=1)

    # plot loss history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(save_dir, "loss_history.png"))
    plt.show()

