import numpy as np
from sklearn.decomposition import NMF


class TimeVaryingSynergy:
    """Time-varying synergies.
    """

    def __init__(self, n_synergies, synergy_length):
        """
        Args:
            n_synergies: Number of synergies
        """
        self.n_synergies = n_synergies
        self.synergy_length = synergy_length

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.data_length = None

    def extract(self, data, max_iter=1000):
        """Extract time-varying synergies from given data.

        Data is assumed to have the shape (#data, data-length, #DoF).
        Synergies have the shape (#synergies, synergy-length, #DoF).
        """
        # Get shape information
        self.dof = data.shape[2]

        # Not implemented
        self.synergies = np.empty((self.n_synergies, self.synergy_length, self.dof))

        return self.synergies

    def encode(self, data, max_iter=1000):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, data-length, #DoF).
        Synergy activities are represented a tuple of the amplitude and delay; both have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Not implemented
        amplitude = np.empty((data.shape[0], self.n_synergies))
        delays = np.empty((data.shape[0], self.n_synergies))
        activities = (amplitude, delays)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, data-length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Not implemented
        data = np.empty((self.dof, self.data_length))

        return data
