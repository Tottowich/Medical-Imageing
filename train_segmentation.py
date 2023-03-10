import os
import numpy as np
np.random.seed(2023)  # Set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(2023)
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from data.data_generator import DataGenerator, DataGeneratorAutoEncoder, DataGeneratorMaskLabel
from data.augmentor import Augmentor
from archs.segmentation.unet import build_unet
from training_utils import AutoEncoderTrainer
# Learning rate schedule
from tensorflow.keras.callbacks import LearningRateScheduler
# Adaptive learning rate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Metrics
from tensorflow.keras.metrics import MeanIoU
from callbacks import ShowSegmentationPredictionCallback

import matplotlib.pyplot as plt
if __name__=="__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU(s) available (using '{gpus[0].name}'). Training will be lightning fast!")
    else:
        print("No GPU(s) available. Training will be suuuuper slow!")

    # NOTE: These are the packages you will need for the assignment.
    # NOTE: You are encouraged to use the course virtual environment, which already has GPU support.
    gen_dir = "data/"  # Change if you have copied the data locally on your machine
    array_labels = ['t1', 't1ce', 't2', 'flair']  # Available arrays are: 't1', 't1ce', 't2', 'flair', 'mask'.
    N_CLASSES = 2  # Number of classes in the segmentation task
    batch_size = 16
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    augmentor = Augmentor()
    gen_train = DataGeneratorMaskLabel(data_path=gen_dir + 'training',
                            arrays=array_labels,
                            batch_size=batch_size,augmentor=augmentor)
    gen_val = DataGeneratorMaskLabel(data_path=gen_dir + 'validating',
                            arrays=array_labels,
                            batch_size=batch_size)

    gen_test = DataGeneratorMaskLabel(data_path=gen_dir + 'testing',
                            arrays=array_labels,
                            batch_size=batch_size)
    print(f"Number of samples in training set: {gen_train.n_samples}")
    print(f"Number of samples in validation set: {gen_val.n_samples}")
    print(f"Number of samples in testing set: {gen_test.n_samples}")
    # NOTE: What arrays are you using? Their order will be the same as their unpacking order during training!
    # NOTE: What batch size are you using? Should you use more? Or less?
    # NOTE: Are you using the correct generators for the correct task? Training for training and validating for validating?
    for idx,(x,y) in enumerate(gen_train):
        print('x shape: ',np.shape(x))
        print('y shape: ',np.shape(y))
        if idx==0:
            break
    H = 256
    W = 256
    C = 1
    # Training parameters
    EPOCHS = 20
    MODEL_PATH = "models/"
    NAME = "unet_segmentation_scratch"
    pretrained = False
    pretrained_path = "models/unet_autoencoder_deeper/weights/epoch_10"
    initial_learning_rate = 0.01
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # Segmentation loss - SparseCategoricalCrossentropy
    # Dice Loss:
    def dice_loss(y_true, y_pred, smooth=1):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))
    # Custom loss function:
    binary_crossentropy = keras.losses.BinaryCrossentropy()
    def custom_loss(y_true, y_pred):
        return 0.75*dice_loss(y_true, y_pred) + 0.25*binary_crossentropy(y_true, y_pred)
    loss = custom_loss
    metrics = [keras.metrics.BinaryAccuracy()]
    save_metric = "val_loss"
    save_best_only = False
    save_interval = 15
    scheduler = ReduceLROnPlateau(monitor=save_metric, factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.00001)
    early_stopping = EarlyStopping(monitor=save_metric, min_delta=0.00001, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    show_predictions = ShowSegmentationPredictionCallback(gen_val)
    callbacks = [scheduler, early_stopping]#, show_predictions]
    if not pretrained:
        # Model parameters
        input_shape = (H,W,C)
        num_classes = C # Number of classes is equal to the number of channels in the output
        filters = [8,16,32,64,128,256]
        kernel_size = 3
        strides = 1
        padding = "same"
        activation = "relu"
        drop_rate_encoder = [0.01,0.2,0.2,0.1,0]
        drop_rate_decoder = [0.0,0.0,0]
        depth_encoder = [1,1,2,3]#[3,3,4,5]
        depth_decoder = [2,1,1,0]#[3,2,1,1,0]
        output_depth = 2
        output_activation = "sigmoid"

        unet = build_unet(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            depth_encoder=depth_encoder,
            depth_decoder=depth_decoder,
            drop_rate_encoder=drop_rate_encoder,
            drop_rate_decoder=drop_rate_decoder,
            output_depth=output_depth,
            output_activation=output_activation,
        )
    else:
        print(f"Loading pretrained model from {pretrained_path}...")
        unet = load_model(pretrained_path)
    unet.summary()
    trainer = AutoEncoderTrainer(
        model=unet,
        epochs=EPOCHS,
        model_path=MODEL_PATH+NAME,
        optimizer=optimizer,
        loss_fn=loss,
        train_generator=gen_train,
        val_generator=gen_val,
        test_generator=gen_test,
        metrics=metrics,
        save_metric=save_metric,
        save_interval=save_interval,
        callbacks=callbacks,)
    gen_train.plot_examples(2)
    gen_val.plot_examples(2)
    trainer.train()




