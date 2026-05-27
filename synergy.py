import numpy as np
from sklearn.decomposition import PCA, NMF
import torch


class SpatialSynergy:
    """Spatial synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method PCA or NMF
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None

    def extract(self, data, max_iter=1000):
        """Extract spatial synergies from given data.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergies have the shape (#synergies, #DoF).
        """
        # Get shape information
        self.dof = data.shape[-1]

         # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)  # transform data from dof to 2*dof dimensions
            self.dof = data.shape[2]  # Update the number of DoF

        # Reshape given data
        data = data.reshape((-1, self.dof))

        if self.method == "nmf" or self.method == "negative-nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)    #used to learn the transformation from data to synergies
            self.synergies = self.model.components_
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies) #reduce the dimensionality of the data from dof to n_synergies
            self.model.fit(data)
            self.synergies = self.model.components_

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, length, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        data_shape = data.shape

          # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape given data from a 3D array to a 2D array, the -1 infers the size of that dimension, knowing the size of the other dimensions
        data = data.reshape((-1, self.dof))  # shape: (#trajectories * length, #DoF)

        # Encode the data
        activities = self.model.transform(data)

        # Reshape activities
        activities = activities.reshape((data_shape[0], data_shape[1], self.n_synergies))  # shape: (#trajectories, length, #synergies)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities have the shape (#trajectories, length, #synergies).
        Data is assumed to have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Keep the shape temporarily
        act_shape = activities.shape

        # Reshape given activities
        activities = activities.reshape((-1, self.n_synergies))  # shape: (#trajectories * length, #synergies)

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)

        # Reshape reconstruction data
        data = data.reshape((act_shape[0], act_shape[1], self.dof))  # shape: (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data



class SpatioTemporalSynergy:
    """Spatio-temporal synergies.
    """

    def __init__(self, n_synergies, method="nmf"):
        """
        Args:
            n_synergies: Number of synergies
            method: Synergy extraction method pca, nmf, or negative-nmf
        """
        self.n_synergies = n_synergies
        self.method = method

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.length = None

    def extract(self, data, max_iter=100000):
        """Extract spatio-temporal synergies from given data.

        Data is assumed to have the shape (#data, length, #DoF).
        Synergies have the shape (#synergies, length, #DoF).
        """
        # Get shape information
        self.length = data.shape[1]
        self.dof = data.shape[2]

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)  # transform data from dof to 2*dof dimensions
            self.dof = data.shape[2]  # Update the number of DoF

        # Reshape given data
        data = data.reshape((data.shape[0], -1))  # shape: (#data, length * #DoF)

        if self.method == "nmf" or self.method == "negative-nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies)
            self.model.fit(data)
            self.synergies = self.model.components_
            self.synergies = self.synergies.reshape((self.n_synergies, self.length, self.dof))  # Reshape synergies

        return self.synergies

    def encode(self, data):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, length, #DoF).
        Synergy activities have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Convert the data to non-negative signals
        if self.method == "negative-nmf":
            data = transform_nonnegative(data)

        # Reshape the data from (#trajectories, length, #DoF) to (#trajectories, length * #DoF)
        data = data.reshape((-1, self.length*self.dof))

        # Encode the data
        activities = self.model.transform(data)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Decode the synergy activities
        data = self.model.inverse_transform(activities)
        data = data.reshape((-1, self.length, self.dof))  # Reshape the shape from (#trajectories, length * #DoF) to (#trajectories, length, #DoF)

        # Convert non-negative signals backwards
        if self.method == "negative-nmf":
            data = inverse_transform_nonnegative(data)

        return data
    

    

def transform_nonnegative(data):
    """Convert a data that has negative values to non-negative signals with doubled dimensions.

    Data is assumed to have the shape (#trajectories, length, #DoF).
    Converted non-negative data have the shape (#trajectories, length, 2 * #DoF).
    """
    n_dof = data.shape[2]  # Dimensionality of the original data

    # Convert the data to non-negative signals
    data_nn = np.empty((data.shape[0], data.shape[1], n_dof*2))
    data_nn[:, :, :n_dof] = +np.maximum(data, 0.0)
    data_nn[:, :, n_dof:] = -np.minimum(data, 0.0)

    return data_nn

def inverse_transform_nonnegative(data):
    """Inverse conversion of `transform_nonnegative()`; Convert non-negative signals to a data that has negative values.

    Non-negative data is assumed to have the shape (#trajectories, length, 2 * #DoF).
    Reconstructed data have the shape (#trajectories, length, #DoF).
    """
    n_dof = int(data.shape[2] / 2)  # Dimensionality of the original data ([2] refers to the last dimension, which is the third == [-1])

    # Restore the original data
    data_rc = np.empty((data.shape[0], data.shape[1], n_dof))
    data_rc = data[:, :, :n_dof] - data[:, :, n_dof:] #the property written in the paper, a = a+(from 0-dof-1) - a- (from dof to 2*dof-1) 

    return data_rc

def R2(x, y):
    e = x - y
    v = x - np.mean(x)
    fvu = np.sum(e**2) / np.sum(v**2)
    return 1 - fvu
