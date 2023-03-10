# Training utilities

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Optimizer
import tensorboard as tb
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import os
import datetime
import time
import sys
import glob
import shutil
import random
from data.data_generator import DataGenerator, DataGeneratorMaskLabel, DataGeneratorAutoEncoder
from data.augmentor import Augmentor
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional

def create_model_directory(model_name:str, model_dir:str="models/")->str:
    """Create a directory for the model and return the path"""
    # Create a directory for the model. Containing, logs, weights, and plots
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        # Make subdirectories for logs, weights, and plots
        os.makedirs(os.path.join(model_path, "logs"))
        os.makedirs(os.path.join(model_path, "weights"))
        os.makedirs(os.path.join(model_path, "plots"))
    else:
        print("Model directory already exists")
    return model_path

# class Augmentor:
#     """Class used for augmentations"""
#     def __init__(self):
#         pass # TODO: Implement augmentation
#     def ___call__(self, x,y):
#         return (x,y)
#     @property
#     def augmentations(self)->List:
#         return []

class Trainer:
    """
    This class is used to keep track of the training run.
    It contains the model, the model path, the tensorboard callback and, the training history.
    """
    def __init__(self, 
                model:Model,
                epochs:int=1,
                model_path:str=None,
                optimizer:Optimizer=None,
                loss_fn:str=None,
                train_generator:DataGenerator=None,
                val_generator:DataGenerator=None,
                test_generator:DataGenerator=None,
                augmentor:"Augmentor"=None,
                save_interval:int=1,
                metrics:List[str]=None,
                save_metric:str = "val_loss",
                tensorboard_callback:tb=None,
                callbacks:List=None):
        self.model = model
        assert model_path is not None, f"Model path must be specified! Got '{model_path}'."
        assert train_generator is not None, f"Training generator must be specified! Got '{train_generator}'."
        self.train_generator = train_generator
        self.batch_size = self.train_generator.batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.augmentor = augmentor
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self._initialize_directories()
        self.epoch = 0
        self.epochs = epochs
        self.metrics = metrics
        self.save_metric = save_metric
        self.save_interval = save_interval
        self._callbacks = callbacks
        if tensorboard_callback is not None:
            self.tensorboard_callback = tensorboard_callback
        else:
            self._initialize_tensorboard()
        self._initialize_checkpoint()
        self.compile_model()
    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)
    def _initialize_directories(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            # Make subdirectories for logs, weights, and plots
            os.makedirs(os.path.join(self.model_path, "logs"))
            os.makedirs(os.path.join(self.model_path, "weights"))
            os.makedirs(os.path.join(self.model_path, "plots"))
        else:
            raise ValueError(f"Model directory already exists: {self.model_path}")
    def _save_intervals(self):
        """Return the number of epochs between each save"""
        # Get number of batches between each save
        batches_per_save = self.save_interval*len(self.train_generator) // self.batch_size
        return batches_per_save
    def _initialize_tensorboard(self):
        """Initialize tensorboard callback"""
        log_dir = os.path.join(self.model_path, "logs")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Set the model to the tensorboard callback since training will be do in a custom loop
        self.tensorboard_callback.set_model(self.model)
        # Create the writers for the present data generators
        self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        if self.val_generator is not None:
            self.val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))
        if self.test_generator is not None:
            self.test_writer = tf.summary.create_file_writer(os.path.join(log_dir, "test"))
    def _initialize_checkpoint(self):
        """Initialize checkpoint callback"""
        filepath = self._save_path()
        self.checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, monitor=self.save_metric, save_best_only=False, save_weights_only=False, mode='auto', save_freq=int(self._save_intervals()))
    @property
    def callbacks(self)->List:
        """Return the callbacks"""
        return self._callbacks+[self.tensorboard_callback, self.checkpoint_callback]
    def _write_tensorboard(self, writer, logs:Dict):
        """Write the logs to tensorboard"""
        with writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=self.epoch)
    def open_tensorboard(self):
        """Open tensorboard in the browser"""
        log_dir = os.path.join(self.model_path, "logs")
        os.system(f"tensorboard --logdir={log_dir}")
    def _save_path(self):
        """Return the path to save the model"""
        return self.model_path + "/weights/epoch_{epoch:02d}/"
    def save_model(self, model_name:str=None,best:bool=False):
        """Save the model"""
        if model_name is None:
            model_name = self.model_name
        if best:
            model_name = f"best_{model_name}"
        else:
            model_name = f"epoch_{self.epoch}_{model_name}"
        model_path = os.path.join(self.model_path, "weights", model_name)
        self.model.save(model_path)
    def train(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    def evaluate(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    def get_model(self):
        return self.model
    
class ClassifierTrainer(Trainer):
    """Trainer for the classifier"""
    def __init__(self, # Same as Trainer
                model:Model,
                epochs:int=1,
                model_path:str=None,
                optimizer:Optimizer=None,
                train_generator:DataGenerator=None,
                val_generator:DataGenerator=None,
                test_generator:DataGenerator=None,
                augmentor:Augmentor=None,
                save_interval:int=1,
                metrics:List[str]=None,
                save_metric:str="val_loss",
                tensorboard_callback:tb=None,
                **kwargs):
        super().__init__(model, epochs, model_path, optimizer, train_generator, val_generator, test_generator, augmentor, save_interval, metrics,save_metric,tensorboard_callback,**kwargs)
    def train(self):
        n_classes = 4
        batches = len(self.train_generator)
        t1_label = tf.one_hot(np.repeat(0, self.train_generator.batch_size), n_classes)
        t1ce_label = tf.one_hot(np.repeat(1, self.train_generator.batch_size), n_classes)
        t2_label = tf.one_hot(np.repeat(2, self.train_generator.batch_size), n_classes)
        flair_label = tf.one_hot(np.repeat(3, self.train_generator.batch_size), n_classes)
        labels = np.concatenate((t1_label,t1ce_label,t2_label, flair_label), axis=0)

        for epoch in range(1,self.epochs+1):
            training_loss = [] # Track loss per epoch.
            validating_loss = []
            pbar = tqdm(enumerate(self.train_generator)) # Progess bar to make it less boring, and trackable.
            # Training data
            for idx, (t1, t1ce,t2,flair) in pbar:
                print(idx)
                if (idx+1)%10==0:
                    pbar.set_description(f"Training Epoch {epoch}/{self.epochs}. {idx+1}/{batches} Batch. Training Loss: {h[0]:.3e}, Accuracy: {h[1]:.3e}")
                images = np.concatenate((t1, t1ce, t2, flair), axis=0)
                h = self.model.train_on_batch(images, labels)
                training_loss.append(h)
                
            train_vals = np.array(training_loss) # Convert to numpy for faster computation.
            ave_train_loss = train_vals[:,0].mean() # Get average loss and accuracy over the epoch.
            ave_train_acc =  train_vals[:,1].mean()
            with self.train_writer.as_default():
                tf.summary.scalar("loss", ave_train_loss, step=self.epoch)
                tf.summary.scalar("accuracy", ave_train_acc, step=self.epoch)
            pbar.set_description(f"Training Epoch {epoch+1}/{self.epochs}. {idx+1}/{batches} Batches. Training Loss: {ave_train_loss:.3e}, Accuracy: {ave_train_acc:.3e}")
            # Validation data
            for idx, (t1, t1ce,t2,flair) in enumerate(self.val_generator):
                images = np.concatenate((t1, t1ce, t2, flair), axis=0)
                validating_loss.append(self.model.test_on_batch(images, labels)[-1])
            val_acc = np.mean(validating_loss)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(best=True)
            elif not (epoch+1)%self.save_interval:
                self.save_model()
            with self.val_writer.as_default():
                tf.summary.scalar("accuracy", val_acc, step=self.epoch)
            
            self.epoch += 1
            print(f"Epoch: {epoch + 1:2d}. Average accuracy - Training: {ave_train_acc:.3e}, Validation: {np.mean(validating_loss):.3e}")
        self.save_model()
    def test(self):
        """Test the model on the test data"""
        testing_loss = []
        n_classes = 4
        t1_label = tf.one_hot(np.repeat(0, self.train_generator.batch_size), n_classes)
        t1ce_label = tf.one_hot(np.repeat(1, self.train_generator.batch_size), n_classes)
        t2_label = tf.one_hot(np.repeat(2, self.train_generator.batch_size), n_classes)
        flair_label = tf.one_hot(np.repeat(3, self.train_generator.batch_size), n_classes)
        labels = np.concatenate((t1_label,t1ce_label,t2_label, flair_label), axis=0)
        for idx, (t1, t1ce,t2,flair) in enumerate(self.test_generator):
            images = np.concatenate((t1, t1ce, t2, flair), axis=0)
            testing_loss.append(self.model.test_on_batch(images, labels)[-1])
        test_acc = np.mean(testing_loss)
        with self.test_writer.as_default():
            tf.summary.scalar("accuracy", test_acc, step=self.epoch)
        print(f"Test accuracy: {test_acc:.3e}")
        return test_acc

class AutoEncoderTrainer(Trainer):
    """Trainer for the autoencoder"""
    def __init__(self, # Same as Trainer
                model:Model,
                epochs:int=1,
                model_path:str=None,
                optimizer:Optimizer=None,
                loss_fn:str="binary_crossentropy",
                train_generator:DataGeneratorAutoEncoder=None,
                val_generator:DataGeneratorAutoEncoder=None,
                test_generator:DataGeneratorAutoEncoder=None,
                augmentor:Augmentor=None,
                save_interval:int=1,
                metrics:List[str]=None,
                save_metric:str="val_loss",
                tensorboard_callback:tb=None,
                **kwargs):
        super().__init__(model, epochs, model_path, optimizer, loss_fn, train_generator, val_generator, test_generator, augmentor, save_interval,metrics,save_metric, tensorboard_callback, **kwargs)
    def train(self):
        self.compile_model()
        # Open tensorboard
        history = self.model.fit(self.train_generator, epochs=self.epochs, validation_data=self.val_generator, callbacks=self.callbacks)
        return history

if __name__ == "__main__":
    # Create the trainer
    trainer = AutoEncoderTrainer()
    trainer.open_tensorboard()

