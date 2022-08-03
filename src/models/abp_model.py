import os
#import tensorflow as tf

#from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
#from keras.backend import repeat_elements, expand_dims
#from tensorflow.keras.initializers import RandomUniform, RandomNormal, Constant
#from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
#from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, ZeroPadding1D, UpSampling1D
#from tensorflow.keras.layers import Lambda, Activation, Concatenate, Softmax
#from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, SeparableConv1D, Conv2DTranspose
#from tensorflow.keras.layers import Input, Reshape, Cropping1D, SpatialDropout1D, RepeatVector, ReLU, Activation
#from tensorflow.keras.layers import Add
#from tensorflow.keras.layers import Bidirectional, LSTM
#from tensorflow.keras import regularizers
#from tensorflow.keras.utils import plot_model
#from tensorflow.keras import backend as K

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Nadam
from keras.backend import repeat_elements, expand_dims
from keras.initializers import RandomUniform, RandomNormal, Constant
from keras.losses import mean_squared_error, mean_absolute_error
from keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, ZeroPadding1D, UpSampling1D
from keras.layers import Lambda, Activation, Concatenate, Softmax
from keras.layers import Conv1D, MaxPool1D, Flatten, SeparableConv1D, Conv2DTranspose
from keras.layers import Input, Reshape, Cropping1D, SpatialDropout1D, RepeatVector, ReLU, Activation
from keras.layers import Add
from keras.layers import Bidirectional, LSTM
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import src.project_configs as project_configs

# def batch_custom_loss(batch_size):
#     def custom_loss(y_true, y_pred, lambda_param=0.80):
#         loss = []
#         for i in range(batch_size):
#             #for y_t, y_p in zip(y_true, y_pred):
#             # multiply ABP with mask to get sys/dias BP values
#             true_bp_values = y_true[i][:,0] * y_true[i][:,1]
#             # multiply pred ABP with mask to get pred sys/dias BP values
#             pred_bp_values = y_pred[i][:,0] * y_true[i][:,1]
#             # calculate mean absolute distance between true/pred values
#             #bp_error = mean_absolute_error(true_bp_values, pred_bp_values)
#             bp_error = mean_squared_error(true_bp_values, pred_bp_values)
#             # get mean absolute error between all true/pred values
#             #mae = mean_squared_error(y_true[i], y_pred[i])
#             mae = mean_squared_error(y_true[i][:,0], y_pred[i][:,0])
#             #loss.append(lambda_param*mae + (1.-lambda_param)*bp_error)
#             loss.append(mae)
# #             loss.append(mae + bp_error)
#         return K.variable(value=np.array(loss))
#         # return tf.convert_to_tensor(loss)
#     return custom_loss

def batch_custom_loss(y_true, y_pred):
    #for y_t, y_p in zip(y_true, y_pred):
    # multiply ABP with mask to get sys/dias BP values
    true_bp_values = y_true[:, :, 0] * y_true[:, :, 1]
    # # # multiply pred ABP with mask to get pred sys/dias BP values
    pred_bp_values = y_pred[:, :, 0] * y_true[:, :, 1]
    # # # calculate mean absolute distance between true/pred values
    # # #bp_error = mean_absolute_error(true_bp_values, pred_bp_values)
    bp_error = mean_squared_error(true_bp_values, pred_bp_values)
    # get mean absolute error between all true/pred values
    #mae = mean_squared_error(y_true[i], y_pred[i])
    mse = mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0])
    #loss.append(lambda_param*mae + (1.-lambda_param)*bp_error)
    return mse + bp_error


def create_model(trainable=True, save_dir=None):

    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 8

    def res_block_1(inputY, num_filters=128, kernel_size=7, trainable=True):
        # output = Conv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None, kernel_regularizer=reg, bias_regularizer=reg)(inputs)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(inputY)
        outputY = BatchNormalization(epsilon=batch_norm_epsilon)(outputY)
        outputY = LeakyReLU(alpha=0.0)(outputY)
        #    output = Dropout(rate=dropout_rate)(outputY)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputY)
        # this returns x + y.
        outputY = add([outputY, inputY])
        return outputY

    def res_block_2(inputX, num_filters=128, kernel_size=5, trainable=True):
        print('inputX shape:', inputX.shape)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(inputX)
        print("oXs:", outputX.shape)
        outputX = LeakyReLU(alpha=0.0)(outputX)
        outputX = SpatialDropout1D(rate=dropout_rate)(outputX)
#        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
#                                  activation=None, trainable=trainable)(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(outputX)
        outputX = LeakyReLU(alpha=0.0)(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
#        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
#                                  activation=None, trainable=trainable)(outputX)

        #     output1 = SeparableConv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None)(output)
        #     output1 = BatchNormalization()(output1)
        #     output1 = LeakyReLU(alpha=0.2)(output1)
        #     output1 = SeparableConv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None)(output1)

        #     output2 = SeparableConv1D(num_filters, kernel_size=kernel_size-2, padding='same', activation=None)(output)
        #     output2 = BatchNormalization()(output2)
        #     output2 = LeakyReLU(alpha=0.2)(output2)
        #     output2 = SeparableConv1D(num_filters, kernel_size=kernel_size-2, padding='same', activation=None)(output2)

        #     output3 = SeparableConv1D(num_filters, kernel_size=kernel_size-4, padding='same', activation=None)(output)
        #     output3 = BatchNormalization()(output3)
        #     output3 = LeakyReLU(alpha=0.2)(output3)
        #     output3 = SeparableConv1D(num_filters, kernel_size=kernel_size-4, padding='same', activation=None)(output3)
        # this returns x + y.
        print("o2s", outputX.shape)
        outputX = add([outputX, inputX])
        return outputX

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_channels),
                   batch_shape=(project_configs.batch_size, project_configs.window_size + 2 * project_configs.padding_size, num_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 1:])(inputs)
    pleth_tensor = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, pleth_index])(inputs)
    print("wave tensors shape:", wave_tensors.shape)
    nibp_vals = Lambda(lambda x: x[:, project_configs.window_size - 1, -num_static_vars:])(inputs)
    print("nibp vals shape:", nibp_vals.shape)
    nibp_vals = RepeatVector(project_configs.window_size)(nibp_vals)
    print("nibp shape:", nibp_vals.shape)
    # mean_abp_val = Lambda(lambda x: x[:,length-1,-5])(inputs)
    mean_abp_val = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, mean_abp_index])(inputs)
    mean_abp_val = Reshape((-1, 1))(mean_abp_val)

    # shift pleth wave using mean ABP
    pleth_tensor_shifted = add([pleth_tensor, mean_abp_val])
    pleth_tensor_shifted = Reshape((-1, 1))(pleth_tensor_shifted)
    print(pleth_tensor_shifted.shape)

    ####output = ZeroPadding1D(padding=padding_size)(inputs)
    output1 = Conv1D(int(project_configs.num_filters/2), kernel_size=9, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size,
                                           num_channels),
                              trainable=trainable)(wave_tensors)
    output1 = BatchNormalization(epsilon=batch_norm_epsilon)(output1)
    output1 = LeakyReLU(alpha=0.0)(output1)

    output2 = Conv1D(int(project_configs.num_filters/2), kernel_size=5, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size,
                                           num_channels),
                              trainable=trainable)(wave_tensors)
    output2 = Cropping1D(cropping=(2, 2))(output2)
    output2 = BatchNormalization(epsilon=batch_norm_epsilon)(output2)
    output2 = LeakyReLU(alpha=0.0)(output2)

    # output1 = SeparableConv1D(64, kernel_size=64, strides=64, activation=None, input_shape=(batch_size, length+2*padding_size, num_channels))(wave_tensors)
    # output1 = BatchNormalization()(output1)
    # output1 = LeakyReLU(alpha=0.2)(output1)
    # print("o1", output1.shape)

    # output2 = SeparableConv1D(64, kernel_size=256, strides=int(100/2), activation=None, input_shape=(batch_size, length+2*padding_size, num_channels))(wave_tensors)
    # # output2 = Cropping1D(cropping=(2, 2))(output2)
    # output2 = BatchNormalization()(output2)
    # output2 = LeakyReLU(alpha=0.2)(output2)
    # print("o2", output2.shape)
    # output2 = SeparableConv1D(64, kernel_size=5, activation=None)(output2)
    # # output2 = Cropping1D(cropping=(2, 2))(output2)
    # output2 = BatchNormalization()(output2)
    # output2 = LeakyReLU(alpha=0.2)(output2)

    output = Concatenate(axis=-1)([output1, output2])

    for i in range(1):
        output = res_block_1(output, num_filters=project_configs.num_filters, kernel_size=16, trainable=trainable)

    # output = MaxPool1D()(output)
    # for i in range(2):
    #     output = res_block_2(output, num_filters=256, kernel_size=256)
    # # output = MaxPool1D()(output)
    # for i in range(2):
    #     output = res_block_2(output, num_filters=256, kernel_size=256)
    # output = MaxPool1D()(output)
    for i in range(2):
        output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size=32, trainable=trainable)
    for i in range(2):
        output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size=64, trainable=trainable)

    # output = add([output, mean_abp_val])
    # for i in range(1):
    # output = res_block_2(output, num_filters=int(output.shape[-1]), kernel_size=8)

    output = add([output, mean_abp_val])

    # output = MaxPool1D()(output)
    # output = Flatten()(output)
    #### added for concatenating non-invasive blood pressure measurements
    # print("pre concat shape:", output.shape)
    # output = Concatenate(axis=-1)([output, nibp_vals])
    # print("post concat shape:", output.shape)
    ####

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = Activation('relu')(output)
    output = Conv1D(int(project_configs.num_filters/2), kernel_size=32, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=True)(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = Activation('relu')(output)

    # nibp_vals = SpatialDropout1D(rate=dropout_rate)(nibp_vals)
    # mean_abp_val = Conv1D(1, kernel_size=1, use_bias=True, bias_initializer=Constant(c), kernel_regularizer=regularizers.l2(0.01), padding='same', activation=None)(mean_abp_val)
    # mean_abp_val = BatchNormalization()(mean_abp_val)
    # mean_abp_val = Activation('relu')(mean_abp_val)
    # output = add([output, mean_abp_val])

    wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, name="wave")(output)

    bp = res_block_2(wave, num_filters=256, kernel_size=64)
    bp = MaxPool1D()(bp)
    bp = Flatten()(bp)
    bp = BatchNormalization()(bp)
    bp = LeakyReLU(alpha=0.2)(bp)
    bp = Dense(64, activation=None, kernel_regularizer=reg, trainable=True)(bp)
    bp = BatchNormalization()(bp)
    bp = LeakyReLU(alpha=0.2)(bp)
    bp = Dense(3, activation=None, kernel_regularizer=reg, trainable=True, name="bp")(bp)

    # print("pre concat shape:", output.shape)
    # output = Concatenate(axis=-1)([output, nibp_vals])
    # print("post concat shape:", output.shape)

    # output = Dropout(dropout_rate)(output)
    # output = SpatialDropout1D(rate=dropout_rate)(output)
    # output = Dense(512, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None, kernel_regularizer=reg)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)

    # output = Dense(1, activation=None, kernel_initializer=RandomUniform(), use_bias=False, kernel_regularizer=reg)(output)
    # output = add([output, mean_abp_val])
    # output = ReLU(max_value=10.0, threshold=-10.0)(output)
    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave, bp])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss(project_configs.batch_size), "bp": 'mean_squared_error'},
                  loss_weights={"wave": 0.9999, "bp": 0.0001},
                  metrics=['mse'],
                  optimizer="Nadam")
    K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_v2(trainable=True, save_dir=None):

    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    def res_block_1(inputY, num_filters=128, kernel_size=7, trainable=True):
        # output = Conv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None, kernel_regularizer=reg, bias_regularizer=reg)(inputs)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(inputY)
        outputY = BatchNormalization(epsilon=batch_norm_epsilon)(outputY)
        outputY = LeakyReLU(alpha=0.0)(outputY)
        #    output = Dropout(rate=dropout_rate)(outputY)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputY)
        # this returns x + y.
        outputY = add([outputY, inputY])
        return outputY

    def res_block_2(inputX, num_filters=128, kernel_size_layer_1=16, kernel_size_layer_2=128, trainable=True):
        print('inputX shape:', inputX.shape)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(inputX)
        print("oXs:", outputX.shape)
        outputX = ReLU()(outputX)
        outputX = SpatialDropout1D(rate=dropout_rate)(outputX)
#        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
#                                  activation=None, trainable=trainable)(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size_layer_1, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(outputX)
        outputX = ReLU()(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size_layer_2, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        # this returns x + y.
        print("o2s", outputX.shape)
        outputX = add([outputX, inputX])
        return outputX

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_channels),
                   batch_shape=(project_configs.batch_size, project_configs.window_size + 2 * project_configs.padding_size, num_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 1:])(inputs)
    pleth_tensor = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, pleth_index])(inputs)
    print("wave tensors shape:", wave_tensors.shape)
    nibp_vals = Lambda(lambda x: x[:, project_configs.window_size - 1, -num_static_vars:])(inputs)
    print("nibp vals shape:", nibp_vals.shape)
    nibp_vals = RepeatVector(project_configs.window_size)(nibp_vals)
    print("nibp shape:", nibp_vals.shape)
    # mean_abp_val = Lambda(lambda x: x[:,length-1,-5])(inputs)
    mean_abp_val = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, mean_abp_index])(inputs)
    mean_abp_val = Reshape((-1, 1))(mean_abp_val)

    # shift pleth wave using mean ABP
    pleth_tensor_shifted = add([pleth_tensor, mean_abp_val])
    pleth_tensor_shifted = Reshape((-1, 1))(pleth_tensor_shifted)
    print(pleth_tensor_shifted.shape)

    ####output = ZeroPadding1D(padding=padding_size)(inputs)

    output = SeparableConv1D(int(project_configs.num_filters), kernel_size=1, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size,
                                           num_channels),
                              trainable=trainable)(wave_tensors)
    output = Cropping1D(cropping=(4, 4))(output)

    for i in range(2):
        output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=16, trainable=trainable)
    
    #for i in range(2):
    #    output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=128, trainable=trainable)
    #for i in range(2):
    #    output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=256, trainable=trainable)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    #output = SeparableConv1D(int(project_configs.num_filters), kernel_size=5, bias_initializer=Constant(c), padding='same',
    #                activation=None, trainable=True)(output)
    #output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    #output = Activation('relu')(output)

    wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, name="wave")(output)
    #wave = Conv1D(1, kernel_size=32, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
    #              trainable=True, name="wave")(output)


    # print("pre concat shape:", output.shape)
    # output = Concatenate(axis=-1)([output, nibp_vals])
    # print("post concat shape:", output.shape)

    # output = Dropout(dropout_rate)(output)
    # output = SpatialDropout1D(rate=dropout_rate)(output)
    # output = Dense(512, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None, kernel_regularizer=reg)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)

    # output = Dense(1, activation=None, kernel_initializer=RandomUniform(), use_bias=False, kernel_regularizer=reg)(output)
    # output = add([output, mean_abp_val])
    # output = ReLU(max_value=10.0, threshold=-10.0)(output)
    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss(project_configs.batch_size)},
                  #loss_weights={"wave": 0.9999, "bp": 0.0001},
                  metrics=['mse'],
                  optimizer="Nadam")
    K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_v3(trainable=True, save_dir=None):

    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    def res_block_1(inputY, num_filters=128, kernel_size=7, trainable=True):
        # output = Conv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None, kernel_regularizer=reg, bias_regularizer=reg)(inputs)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(inputY)
        outputY = BatchNormalization(epsilon=batch_norm_epsilon)(outputY)
        outputY = LeakyReLU(alpha=0.0)(outputY)
        #    output = Dropout(rate=dropout_rate)(outputY)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputY)
        # this returns x + y.
        outputY = add([outputY, inputY])
        return outputY

    def res_block_2(inputX, num_filters=128, kernel_size=5, trainable=True):
        print('inputX shape:', inputX.shape)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(inputX)
        print("oXs:", outputX.shape)
        outputX = ReLU()(outputX)
        outputX = SpatialDropout1D(rate=dropout_rate)(outputX)
#        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
#                                  activation=None, trainable=trainable)(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(outputX)
        outputX = ReLU()(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        # this returns x + y.
        print("o2s", outputX.shape)
        outputX = add([outputX, inputX])
        return outputX

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 5
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_channels),
                   batch_shape=(project_configs.batch_size, project_configs.window_size + 2 * project_configs.padding_size, num_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 1:])(inputs)
    ekg_tensor = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, ekg_index])(inputs)
    ekg_tensor = Reshape((-1, 1))(ekg_tensor)
    pleth_tensor = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, pleth_index])(inputs)
    pleth_tensor = Reshape((-1, 1))(pleth_tensor)
    print("wave tensors shape:", wave_tensors.shape)
    nibp_vals = Lambda(lambda x: x[:, project_configs.window_size - 1, -num_static_vars:])(inputs)
    print("nibp vals shape:", nibp_vals.shape)
    nibp_vals = RepeatVector(project_configs.window_size)(nibp_vals)
    print("nibp shape:", nibp_vals.shape)
    # mean_abp_val = Lambda(lambda x: x[:,length-1,-5])(inputs)
    mean_abp_val = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, mean_abp_index])(inputs)
    mean_abp_val = Reshape((-1, 1))(mean_abp_val)

    ####output = ZeroPadding1D(padding=padding_size)(inputs)

    output_ekg = SeparableConv1D(int(project_configs.num_filters), kernel_size=1, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size),
                              trainable=trainable)(ekg_tensor)
    #output_ekg = Cropping1D(cropping=(4, 4))(output_ekg)

    for i in range(2):
        output_ekg = res_block_2(output_ekg, num_filters=project_configs.num_filters, kernel_size=16, trainable=trainable)

    output_ekg = BatchNormalization(epsilon=batch_norm_epsilon)(output_ekg)
    output_ekg = ReLU()(output_ekg)
    
    output_ekg = SeparableConv1D(1, kernel_size=5, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=True)(output_ekg)
    output_ekg = BatchNormalization(epsilon=batch_norm_epsilon)(output_ekg)
    output_ekg = Activation('relu')(output_ekg)

    # for PPG signal
    output_ppg = SeparableConv1D(int(project_configs.num_filters), kernel_size=1, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size),
                              trainable=trainable)(pleth_tensor)
    #output_ppg = Cropping1D(cropping=(4, 4))(output_ppg)

    for i in range(2):
        output_ppg = res_block_2(output_ppg, num_filters=project_configs.num_filters, kernel_size=16, trainable=trainable)

    output_ppg = BatchNormalization(epsilon=batch_norm_epsilon)(output_ppg)
    output_ppg = ReLU()(output_ppg)
    
    output_ppg = SeparableConv1D(1, kernel_size=5, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=True)(output_ppg)
    output_ppg = BatchNormalization(epsilon=batch_norm_epsilon)(output_ppg)
    output_ppg = Activation('relu')(output_ppg)

    # concatenate signals together
    output = Concatenate(axis=-1)([output_ekg, output_ppg, nibp_vals])
   
    print("Concat output shape:", output.shape)
    output = SeparableConv1D(int(project_configs.num_filters), kernel_size=5, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=True)(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = Activation('relu')(output)

    wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, name="wave")(output)


    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss(project_configs.batch_size)},
                  #loss_weights={"wave": 0.9999, "bp": 0.0001},
                  metrics=['mse'],
                  optimizer="Nadam")
    K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_v4(trainable=True, save_dir=None):

    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    def res_block_1(inputY, num_filters=128, kernel_size_layer_1=7, kernel_size_layer_2=128, trainable=True):
        # output = Conv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None, kernel_regularizer=reg, bias_regularizer=reg)(inputs)
        outputY = Conv1D(num_filters, kernel_size=kernel_size_layer_1, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(inputY)
        outputY = BatchNormalization(epsilon=batch_norm_epsilon)(outputY)
        outputY = LeakyReLU(alpha=0.0)(outputY)
        #    output = Dropout(rate=dropout_rate)(outputY)
        outputY = Conv1D(num_filters, kernel_size=kernel_size_layer_2, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputY)
        # this returns x + y.
        outputY = Add()([outputY, inputY])
        return outputY

    def res_block_2(inputX, num_filters=128, kernel_size_layer_1=16, kernel_size_layer_2=128, trainable=True):
        print('inputX shape:', inputX.shape)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(inputX)
        print("oXs:", outputX.shape)
        outputX = ReLU()(outputX)
        outputX = SpatialDropout1D(rate=dropout_rate)(outputX)
#        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
#                                  activation=None, trainable=trainable)(outputX)
        outputX = Conv1D(num_filters, kernel_size=kernel_size_layer_1, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputX)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(outputX)
        outputX = ReLU()(outputX)
        outputX = Conv1D(num_filters, kernel_size=kernel_size_layer_2, bias_initializer=Constant(c), padding='same',
                                  activation=None, trainable=trainable)(outputX)
        # this returns x + y.
        print("o2s", outputX.shape)
        outputX = Add()([outputX, inputX])
        return outputX

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_channels = 11
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=1.0)
    optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 0:])(inputs)
    # pleth_tensor = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, pleth_index])(inputs)
    # print("wave tensors shape:", wave_tensors.shape)
    # nibp_vals = Lambda(lambda x: x[:, project_configs.window_size - 1, -num_static_vars:])(inputs)
    # print("nibp vals shape:", nibp_vals.shape)
    # nibp_vals = RepeatVector(project_configs.window_size)(nibp_vals)
    # print("nibp shape:", nibp_vals.shape)
    # # mean_abp_val = Lambda(lambda x: x[:,length-1,-5])(inputs)
    # mean_abp_val = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, mean_abp_index])(inputs)
    # mean_abp_val = Reshape((-1, 1))(mean_abp_val)

    # shift pleth wave using mean ABP
    # pleth_tensor_shifted = add([pleth_tensor, mean_abp_val])
    # pleth_tensor_shifted = Reshape((-1, 1))(pleth_tensor_shifted)
    # print(pleth_tensor_shifted.shape)

    ####output = ZeroPadding1D(padding=padding_size)(inputs)

    output = Conv1D(int(project_configs.num_filters), kernel_size=9, bias_initializer=Constant(c), activation=None,
                              input_shape=(project_configs.batch_size,
                                           project_configs.window_size + 2 * project_configs.padding_size,
                                           num_channels),
                              trainable=trainable)(wave_tensors)
    #output = Cropping1D(cropping=(4, 4))(output)

    #for i in range(2):
    #    output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=16, trainable=trainable)
    
    for i in range(1):
        output = res_block_1(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=128, trainable=trainable)
    for i in range(2):
        output = res_block_2(output, num_filters=project_configs.num_filters, kernel_size_layer_1=16, kernel_size_layer_2=256, trainable=trainable)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    #output = SeparableConv1D(int(project_configs.num_filters), kernel_size=5, bias_initializer=Constant(c), padding='same',
    #                activation=None, trainable=True)(output)
    #output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    #output = Activation('relu')(output)

    #wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
    #              trainable=True, name="wave")(output)
    wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, name="wave")(output)


    # print("pre concat shape:", output.shape)
    # output = Concatenate(axis=-1)([output, nibp_vals])
    # print("post concat shape:", output.shape)

    # output = Dropout(dropout_rate)(output)
    # output = SpatialDropout1D(rate=dropout_rate)(output)
    # output = Dense(512, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None, kernel_regularizer=reg)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)
    # output = Dense(128, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = LeakyReLU(alpha=0.2)(output)

    # output = Dense(1, activation=None, kernel_initializer=RandomUniform(), use_bias=False, kernel_regularizer=reg)(output)
    # output = add([output, mean_abp_val])
    # output = ReLU(max_value=10.0, threshold=-10.0)(output)
    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss},
                  metrics=['mse'],
                  optimizer=optimizer)
    # K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_vnet(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1
    num_channels = 32
    if project_configs.window_size == 200:
        num_convolutions = (1, 2, 3)
    else:
        num_convolutions = (1, 2, 3, 3)
    num_levels = len(num_convolutions)
    #num_convolutions = (1, 2, 3, 3, 3)
    bottom_convolutions = 3
    kernel_size = 5

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_input_channels = 6 + 5
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)


    inputs = Input(shape=(400 + 2 * project_configs.padding_size,
                          15))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 0:])(inputs)

    def convolution_block(layer_input, n_convolutions):
        x = layer_input
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])
        for i in range(n_convolutions):
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            if i == n_convolutions - 1:
                x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)
        return x

    def convolution_block_2(layer_input, fine_grained_features, n_convolutions):
        x = Concatenate(axis=-1)([layer_input, fine_grained_features])
        try:
            n_channels = int(layer_input.shape[-1][-1])
        except TypeError:
            n_channels = int(layer_input.shape[-1])
        if n_convolutions == 1:
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            layer_input = BatchNormalization(epsilon=batch_norm_epsilon)(layer_input)
            x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)
            return x

        x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                   bias_initializer=Constant(c))(x)
        x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
        x = ReLU()(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

        for i in range(1, n_convolutions):
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            # x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            if i == n_convolutions - 1:
                layer_input = BatchNormalization(epsilon=batch_norm_epsilon)(layer_input)
                x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)

        return x

    def down_convolution(x, factor, kernel_size):
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])
        x = Conv1D(n_channels*factor, kernel_size=kernel_size, strides=factor, padding='same',
                   bias_initializer=Constant(c))(x)
        return x

    def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
        """
            FROM: https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
            kernel_size: int, size of the convolution kernel
            strides: int, convolution step size
            padding: 'same' | 'valid'
        """
        try:
            k = Lambda(lambda k: K.expand_dims(k, axis=2), output_shape=(input_tensor.shape[-1][-2], 1,
                                                                     input_tensor.shape[-1][-1]))(input_tensor)
        except TypeError:
            k = Lambda(lambda k: K.expand_dims(k, axis=2), output_shape=(input_tensor.shape[-2], 1,
                                                                     input_tensor.shape[-1]))(input_tensor)

        print("ITS:", k.shape)
        print("ITS length", K.ndim(k))
        k = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(k)
        k = Lambda(lambda k: K.squeeze(k, axis=2))(k)
        return k

    def up_convolution(x, factor, kernel_size):
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])
        print("up conv n_channels:", n_channels)
        x = Conv1DTranspose(x, n_channels // factor, kernel_size=kernel_size, padding='same',
                            strides=factor)
        return x

    output = Conv1D(int(16), kernel_size=5, bias_initializer=Constant(c), padding='same',
                             activation=None,
                             input_shape=(project_configs.batch_size,
                                          project_configs.window_size + 2 * project_configs.padding_size,
                                          num_channels),
                             trainable=trainable)(wave_tensors)
    output = Cropping1D(cropping=(4, 4))(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    features = list()
    # compression
    for l in range(num_levels):
        output = convolution_block(output, n_convolutions=num_convolutions[l])
        features.append(output)
        output = down_convolution(output, factor=2, kernel_size=kernel_size)
        output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
        output = ReLU()(output)
        print(output.shape)

    # bottom
    output = convolution_block(output, bottom_convolutions)

    # decompression
    for l in reversed(range(num_levels)):
        f = features[l]
        output = up_convolution(output, factor=2, kernel_size=kernel_size)
        output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
        output = ReLU()(output)

        output = convolution_block_2(output, f, num_convolutions[l])

    wave = Conv1D(1, kernel_size=1, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, activity_regularizer=regularizers.l2(0.0005), name="wave")(output)

    print("FINAL shape:", wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    #model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.compile(loss=batch_custom_loss, metrics=['mse'], optimizer=optimizer)
#     model.compile(loss={"wave": batch_custom_loss},
#                   # loss_weights={"wave": 0.9999, "bp": 0.0001},
#                   metrics=['mse'],
#                   optimizer=optimizer)
    # K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    #plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_lstm(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # reg = regularizers.l2(0.01)
    dropout_rate = 0.3
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_input_channels = 6 + 5
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_input_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 0:])(inputs)

    output = SeparableConv1D(int(16), kernel_size=9, bias_initializer=Constant(c),
                             activation=None,
                             input_shape=(project_configs.batch_size,
                                          project_configs.window_size + 2 * project_configs.padding_size,
                                          num_channels),
                             trainable=trainable)(wave_tensors)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    num_units = 128
    unroll = False

    output = Bidirectional(LSTM(num_units, dropout=dropout_rate, return_sequences=True, unroll=unroll))(output)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = LSTM(num_units, dropout=dropout_rate, return_sequences=True, unroll=unroll)(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output2 = LSTM(num_units, dropout=dropout_rate, return_sequences=True, unroll=unroll)(output)
    output2 = BatchNormalization(epsilon=batch_norm_epsilon)(output2)
    output = Add()([output, output2])
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = LSTM(num_units, dropout=dropout_rate, return_sequences=True, unroll=unroll)(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    # output = LSTM(int(16), dropout=dropout_rate, return_sequences=True, unroll=unroll)(output)
    output = LSTM(1, dropout=dropout_rate, return_sequences=True, unroll=unroll)(output)
    # output = LSTM(3, dropout=dropout_rate, return_sequences=False, unroll=unroll)(output)
    print("LSTM shape:", output.shape)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    wave = Dense(1, activation=None, name="wave")(output)

    # wave = Conv1D(1, kernel_size=32, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
    #               trainable=True, name="wave")(output)

    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    # model.compile(loss='mean_squared_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss(project_configs.batch_size)},
                  # loss_weights={"wave": 0.9999, "bp": 0.0001},
                  metrics=['mse'],
                  optimizer=optimizer)
    # K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_v5(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # reg = regularizers.l2(0.01)
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    def res_block_1(inputY, num_filters=128, kernel_size=7, trainable=True):
        # output = Conv1D(num_filters, kernel_size=kernel_size, padding='same', activation=None, kernel_regularizer=reg, bias_regularizer=reg)(inputs)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                         activation=None, trainable=trainable)(inputY)
        outputY = BatchNormalization(epsilon=batch_norm_epsilon)(outputY)
        outputY = LeakyReLU(alpha=0.0)(outputY)
        #    output = Dropout(rate=dropout_rate)(outputY)
        outputY = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
                         activation=None, trainable=trainable)(outputY)
        # this returns x + y.
        outputY = add([outputY, inputY])
        return outputY

    def res_block_2(inputX, num_filters=128, kernel_size_layer_1=16, kernel_size_layer_2=128, trainable=True):
        print('inputX shape:', inputX.shape)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(inputX)
        print("oXs:", outputX.shape)
        outputX = ReLU()(outputX)
        outputX = SpatialDropout1D(rate=dropout_rate)(outputX)
        #        outputX = Conv1D(num_filters, kernel_size=kernel_size, bias_initializer=Constant(c), padding='same',
        #                                  activation=None, trainable=trainable)(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size_layer_1, bias_initializer=Constant(c),
                                  padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        outputX = BatchNormalization(epsilon=batch_norm_epsilon)(outputX)
        outputX = ReLU()(outputX)
        outputX = SeparableConv1D(num_filters, kernel_size=kernel_size_layer_2, bias_initializer=Constant(c),
                                  padding='same',
                                  activation=None, trainable=trainable, depth_multiplier=depth_multiplier)(outputX)
        # this returns x + y.
        print("o2s", outputX.shape)
        outputX = add([outputX, inputX])
        return outputX

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 6
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    scaled_ppg_index = 1
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.5)
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0, clipnorm=0.5)
    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_channels))
    # inputs2 = Lambda(lambda x: x[:,:,1:])(inputs)
    # inputs = Input(shape=(length, 2), batch_shape=(batch_size, length, 2))
    print(inputs.shape)
    # y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
    # wave_tensors = Lambda(lambda x: x[:,:,pleth_index+1:-num_static_vars])(inputs)
    wave_tensors = Lambda(lambda x: x[:, :, 0:])(inputs)
    scaled_ppg = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, scaled_ppg_index])(
        inputs)
    scaled_ppg = Reshape((-1, 1))(scaled_ppg)

    output = Conv1D(int(project_configs.num_filters), kernel_size=9, bias_initializer=Constant(c),
                 activation=None, padding="same",
                 input_shape=(project_configs.batch_size,
                              project_configs.window_size + 2 * project_configs.padding_size,
                              num_channels),
                 trainable=trainable)(wave_tensors)
    output = Cropping1D(cropping=(4, 4))(output)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    output = Conv1D(int(project_configs.num_filters*2), kernel_size=16, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=trainable)(output)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    output = Conv1D(int(project_configs.num_filters*4), kernel_size=64, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=trainable)(output)

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    output = Conv1D(8, kernel_size=256, bias_initializer=Constant(c), padding='same',
                    activation=None, trainable=trainable)(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    output = Add()([output, scaled_ppg])

    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    wave = Conv1D(1, kernel_size=16, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, name="wave")(output)

    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # model.compile(loss=batch_custom_loss(batch_size), metrics=['mse'], optimizer=optimizer)
    model.compile(loss={"wave": batch_custom_loss(project_configs.batch_size)},
                  # loss_weights={"wave": 0.9999, "bp": 0.0001},
                  metrics=['mse'],
                  optimizer="Nadam")
    # K.set_value(model.optimizer.lr, 0.002)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model


def create_model_sideris(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # reg = regularizers.l2(0.01)
    dropout_rate = 0.5
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    # padding_size = 4
    # num_channels = 9+1+1+1+1
    # num_static_vars = 7
    num_channels = 28
    num_input_channels = 6 + 3
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # default parameters for RMSprop
    #optimizer = RMSprop(lr=0.001, rho=0.9)
    optimizer = RMSprop(lr=0.001, rho=0.9, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_input_channels))
    print(inputs.shape)
    wave_tensors = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, project_configs.ppg_col],
                          output_shape=(project_configs.window_size,))(inputs)
    wave_tensors = Reshape((-1, 1))(wave_tensors)

    num_units = 128
    unroll = False

    output = LSTM(num_units, return_sequences=True, unroll=unroll)(wave_tensors)
    output = Dropout(rate=dropout_rate)(output)
    wave = Dense(1, activation=None, name="wave")(output)

    print(wave.shape)

    model = Model(inputs=inputs, outputs=[wave])
    model.compile(loss={"wave": "mse"},
                  metrics=['mse'],
                  optimizer=optimizer)
    print("LR: {}".format(K.get_value(model.optimizer.lr)))
    # model.compile(loss='mse', metrics=['mse'], optimizer=optimizer)
    model.summary()

    #plot_model(model, to_file=os.path.join(save_dir, "model.png"), show_shapes=True)
    return model
