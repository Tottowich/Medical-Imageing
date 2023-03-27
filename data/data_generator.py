
import tensorflow.keras as keras
from .augmentor import Augmentor
import numpy as np
import os
# class Generator(keras.utils.Sequence):
#     def __init__(self):
#         pass

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 arrays,
                 batch_size=32,
                 augmentor=None,
                 ):

        self.data_path = data_path
        self.arrays = arrays
        self.batch_size = batch_size
        self.augmentor = augmentor

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.arrays)):
                im = npzfile[self.arrays[i]]
                self.in_dims.append((self.batch_size,
                                    *np.shape(im),
                                    self.n_channels))

    def __len__(self):
        """Get the number of batches per epoch."""
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        a = self.__data_generation(list_IDs_temp)
        return a

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        """Generate data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        arrays = []

        for i in range(len(self.arrays)):
            arrays.append(np.empty(self.in_dims[i]).astype(np.single))
        print('arrays shape: ',arrays[0].shape)
        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(len(self.arrays)):
                    x = npzfile[self.arrays[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    x_max = np.max(x)
                    x = x / x_max if x_max > 0 else x
                    arrays[idx][i, ] = x

        return arrays

class DataGeneratorMaskLabel(keras.utils.Sequence):
    def __init__(self,
                 data_path:str,
                 arrays:list,
                 batch_size:int=32,
                 augmentor=None,
                 ):

        self.data_path = data_path
        # Remove Mask from arrays if present
        if 'mask' in arrays:
            arrays.remove('mask')
        self.arrays = arrays
        self.batch_size = batch_size
        self.augmentor = augmentor if augmentor is not None else lambda x,y: (x,y)

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        # Number of files
        # print(f"Number of files: {len(self.file_list)}")
        self.n_formats = len(self.arrays)
        self.n_samples = len(self.file_list)*self.n_formats
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.n_channels = 1 # Input image is always single channel (grayscale)
            im = npzfile[self.arrays[0]]
            self.in_dims = (self.batch_size*self.n_formats,*np.shape(im),self.n_channels)

    def __len__(self):
        """Get the number of batches per epoch."""
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        a,m = self.__data_generation(list_IDs_temp)
        return a,m

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        """Generate data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        arrays = np.empty(self.in_dims).astype(np.single)
        masks = np.empty(self.in_dims).astype(np.single)
        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                y = npzfile['mask'] \
                    .astype(np.single)
                y = np.expand_dims(y, axis=2)
                for idx in range(self.n_formats):
                    x = npzfile[self.arrays[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=2)/255
                    x,y = self.augmentor(x,y)
                    arrays[self.n_formats*i+idx] = x
                    masks[self.n_formats*i+idx] = y
        return arrays, masks
    def get_random(self, n_samples:int=1):
        # Get random choices from the file list
        files = np.random.choice(self.file_list, n_samples)
        # Get the arrays from the files
        return self.__data_generation(files)
    def plot_examples(self, n_examples:int=1):
        import matplotlib.pyplot as plt
        """Plot some examples of the data in two columns."""
        # Get the data
        x, y = self.get_random(n_examples)
        # Plot the data
        fig, ax = plt.subplots(n_examples, 2, figsize=(10, 10))
        for i in range(n_examples):
            # Add titles to the plots
            ax[i, 0].imshow(x[i, :, :, 0], cmap='gray')
            ax[i, 1].imshow(y[i, :, :, 0], cmap='gray')
            ax[i, 0].set_title('Input')
            ax[i, 1].set_title('Label')
        plt.show()
class DataGeneratorAutoEncoder(keras.utils.Sequence):
    def __init__(self,
                 data_path:str,
                 arrays:list,
                 batch_size:int=32,
                 augmentor=None,
                 ):

        self.data_path = data_path
        # Remove Mask from arrays if present
        if 'mask' in arrays:
            arrays.remove('mask')
        self.arrays = arrays
        self.batch_size = batch_size
        self.augmentor = augmentor if augmentor is not None else lambda x,y: (x,y)

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        # Number of files
        # print(f"Number of files: {len(self.file_list)}")
        self.n_formats = len(self.arrays)
        self.n_samples = len(self.file_list)*self.n_formats
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.n_channels = 1 # Input image is always single channel (grayscale)
            im = npzfile[self.arrays[0]]
            self.in_dims = (self.batch_size*self.n_formats,*np.shape(im),self.n_channels)

    def __len__(self):
        """Get the number of batches per epoch."""
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        a,y = self.__data_generation(list_IDs_temp)
        return a,y # Return the same thing twice, image is both input and output

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        """Generate data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        arrays = np.empty(self.in_dims).astype(np.single)
        ys = np.empty(self.in_dims).astype(np.single)
        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(self.n_formats):
                    x = npzfile[self.arrays[idx]] \
                        .astype(np.single)/255
                    x,y = self.augmentor(x,x.copy())
                    x = np.expand_dims(x, axis=2)
                    y = np.expand_dims(y, axis=2)
                    arrays[self.n_formats*i+idx] = x
                    ys[self.n_formats*i+idx] = y
        return arrays,ys
    def get_random(self, n_samples:int=1):
        # Get random choices from the file list
        files = np.random.choice(self.file_list, n_samples)
        # Get the arrays from the files
        return self.__data_generation(files)
    def plot_examples(self, n_examples:int=1):
        import matplotlib.pyplot as plt
        """Plot some examples of the data in two columns."""
        # Get the data
        x, y = self.get_random(n_examples)
        # Plot the data
        fig, ax = plt.subplots(n_examples, 2, figsize=(10, 10))
        for i in range(n_examples):
            # Add titles to the plots
            ax[i, 0].imshow(x[i, :, :, 0], cmap='gray')
            ax[i, 1].imshow(y[i, :, :, 0], cmap='gray')
            ax[i, 0].set_title('Input')
            ax[i, 1].set_title('Label')
        plt.show()


from typing import List, Tuple
def get_generators(gen_type:str,parent_path:str, batch_size:int,array_labels:List[str]=["t1","t1ce","t2","flair"],augmentor=None):
    """Get the generators for the training, validation and testing data."""
    if gen_type == 'autoencoder':
        gen_train = DataGeneratorAutoEncoder(data_path=parent_path + 'training',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=augmentor)
        # gen_train = DataGenerator(data_path=gen_dir + 'training',
        #                         arrays=array_labels,
        #                         batch_size=batch_size)
        gen_val = DataGeneratorAutoEncoder(data_path=parent_path + 'validating',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)

        gen_test = DataGeneratorAutoEncoder(data_path=parent_path + 'testing',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)
    elif gen_type == 'classification':
        gen_train = DataGenerator(data_path=parent_path + 'training',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=augmentor)
        gen_val = DataGenerator(data_path=parent_path + 'validating',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)

        gen_test = DataGenerator(data_path=parent_path + 'testing',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)
    elif gen_type == 'segmentation': # DataGeneratorMaskLabel
        gen_train = DataGeneratorMaskLabel(data_path=parent_path + 'training',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=augmentor)
        gen_val = DataGeneratorMaskLabel(data_path=parent_path + 'validating',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)

        gen_test = DataGeneratorMaskLabel(data_path=parent_path + 'testing',
                                arrays=array_labels,
                                batch_size=batch_size,
                                augmentor=None)


    else:
        raise ValueError('Generator type not recognized.')
    return gen_train, gen_val, gen_test
        