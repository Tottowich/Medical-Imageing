# Tensorflow callbacks for Keras models
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
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
# Metrics
from tensorflow.keras.metrics import MeanIoU

import matplotlib.pyplot as plt

class ShowSegmentationPredictionCallback(Callback):
    def __init__(self, dataset):
        super(ShowSegmentationPredictionCallback, self).__init__()
        self.dataset = dataset
    def on_epoch_begin(self, epoch, logs=None):
        
        # Make a prediction on the first batch of the validation set
        # Retrieve the first batch of the validation from the generator
        x, y = next(iter(self.dataset))
        # Make a prediction on the first batch of the validation set
        y_pred = self.model.predict(x)
        # Retrieve the first image of the batch
        image = x[0]
        # Retrieve the first mask of the batch
        mask = y[0]
        # Retrieve the first predicted mask of the batch
        predicted_mask = y_pred[0]
        # Binary classification
        predicted_mask = np.round(predicted_mask)
        # Plot the first image of the batch
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image[:, :, 0], cmap="gray")
        plt.title("Original image")
        plt.axis("off")
        # Plot the first mask of the batch
        plt.subplot(1, 3, 2)
        plt.imshow(mask[:, :, 0], cmap="gray")
        plt.title("True Mask")
        plt.axis("off")
        # Plot the first predicted mask of the batch
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask[:, :, 0], cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        # Show the plot but do not block the execution
        # Set main title
        plt.suptitle(f"Epoch {epoch}")
        plt.show()

