import os
import numpy as np
np.random.seed(2023)  # Set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(2023)
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
from data_generator import DataGenerator, DataGeneratorAutoEncoder
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
    N_CLASSES = len(array_labels)
    batch_size = 16

    gen_train = DataGeneratorAutoEncoder(data_path=gen_dir + 'training',
                            arrays=array_labels,
                            batch_size=batch_size)
    # gen_train = DataGenerator(data_path=gen_dir + 'training',
    #                         arrays=array_labels,
    #                         batch_size=batch_size)
    gen_val = DataGeneratorAutoEncoder(data_path=gen_dir + 'validating',
                            arrays=array_labels,
                            batch_size=batch_size)

    gen_test = DataGeneratorAutoEncoder(data_path=gen_dir + 'testing',
                            arrays=array_labels,
                            batch_size=batch_size)
    print(f"Number of samples in training set: {gen_train.n_samples}")
    print(f"Number of samples in validation set: {gen_val.n_samples}")
    print(f"Number of samples in testing set: {gen_test.n_samples}")
    # imgs = gen_train[0]
    # for inp in range(np.shape(imgs)[0]):
    #     plt.figure(figsize=(12,5))
    #     for i in range(4):
    #         plt.subplot(1, 4, i + 1)
    #         plt.imshow(imgs[inp][i, :, :, 0], cmap='gray')# Rainbow
    #         plt.title('Image size: ' + str(np.shape(imgs[inp][i, :, :, 0])))
    #         plt.tight_layout()
    #     plt.suptitle('Array: ' + gen_train.arrays[inp])
    #     plt.show()
    # NOTE: What arrays are you using? Their order will be the same as their unpacking order during training!
    # NOTE: What batch size are you using? Should you use more? Or less?
    # NOTE: Are you using the correct generators for the correct task? Training for training and validating for validating?
    for idx,(x,y) in enumerate(gen_train):
        print('x shape: ',np.shape(x))
        print('y shape: ',np.shape(y))
        if idx==0:
            break
    