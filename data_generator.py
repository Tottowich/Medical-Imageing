
import tensorflow.keras as keras
import numpy as np
import os
class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 arrays,
                 batch_size=32,
                 ):

        self.data_path = data_path
        self.arrays = arrays
        self.batch_size = batch_size

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
                    arrays[idx][i, ] = x

        return arrays

class DataGeneratorMaskLabel(keras.utils.Sequence):
    def __init__(self,
                 data_path:str,
                 arrays:list,
                 batch_size:int=32,
                 ):

        self.data_path = data_path
        # Remove Mask from arrays if present
        if 'mask' in arrays:
            arrays.remove('mask')
        self.arrays = arrays
        self.batch_size = batch_size

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
                    x = np.expand_dims(x, axis=2)
                    arrays[self.n_formats*i+idx] = x
                    masks[self.n_formats*i+idx] = y
        return arrays, masks