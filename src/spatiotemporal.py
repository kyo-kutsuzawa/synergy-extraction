import numpy as np
from sklearn.decomposition import PCA, NMF


class SpatioTemporalSynergy:
    """Spatio-temporal synergies.
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
        self.length = None

    def extract(self, data, max_iter=1000):
        """Extract spatio-temporal synergies from given data.

        Data is assumed to have the shape (#data, length, #DoF).
        Synergies have the shape (#synergies, length, #DoF).
        """
        # Get shape information
        self.length = data.shape[1]
        self.dof = data.shape[2]

        # Reshape given data
        data = data.reshape((data.shape[0], -1))  # shape: (#data, length * #DoF)

        if self.method == "nmf":
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

        return data
