import numpy as np
from sklearn.decomposition import PCA, NMF


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

        # Reshape given data
        data = data.reshape((-1, self.dof))

        if self.method == "nmf":
            self.model = NMF(n_components=self.n_synergies, max_iter=max_iter)
            self.model.fit(data)
            self.synergies = self.model.components_
        elif self.method == "pca":
            self.model = PCA(n_components=self.n_synergies)
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

        # Reshape given data
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

        return data
