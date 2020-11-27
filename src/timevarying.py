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


def _generate_example_data(N=3, M=3, T=30, K=2, S=15, plot=True):
    """Check example-data generation code.

    N: Number of data
    M: Number of DoF
    T: Time length of data
    K: Number of synergies
    S: Time length of synergies
    """
    def gaussian(x, mu, std):
        return np.exp(-0.5 * (x - mu)**2 / std**2)

    # Generate synergies
    synergies = np.zeros((K, S, M))
    for k in range(K):
        for m in range(M):
            std = np.random.uniform(S*0.05, S*0.2)
            margin = int(std * 2)
            mu = np.random.randint(margin, S-margin)
            for s in range(S):
                synergies[k, s, m] = gaussian(s, mu, std)

    # Generate synergy activities (i.e., amplitude and delay)
    amplitude = np.random.uniform(0, 1, (N, K))
    delays = np.random.randint(0, T-S, (N, K))

    # Compute a dataset from the synergies and activities
    data = np.zeros((N, T, M))
    for n in range(N):
        for k in range(K):
            ts = delays[n, k]
            data[n, ts:ts+S, :] += amplitude[n, k] * synergies[k, :, :]

    # Plot results if specified
    if plot:
        import matplotlib.pyplot as plt

        # Plot synergy components
        fig = plt.figure()
        fig.suptitle("Synergy components")
        for k in range(K):
            for m in range(M):
                ax = fig.add_subplot(M, K, m*K+k+1)
                ax.plot(np.arange(synergies.shape[1]), synergies[k, :, m])
                ax.set_xlim((0, synergies.shape[1]-1))
                if k == 0:
                    ax.set_ylabel("DoF #{}".format(m+1))
            ax.set_xlabel("synergy #{}".format(k+1))
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Plot reconstruction data
        fig = plt.figure()
        fig.suptitle("Original data")
        axes = [fig.add_subplot(M, 1, m+1) for m in range(M)]
        for n in range(N):
            for m, ax in enumerate(axes):
                ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
                ax.set_xlim((0, data.shape[1]-1))
                ax.set_ylabel("DoF #{}".format(m+1))
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plt.show()

    return data, synergies, (amplitude, delays)


if __name__ == "__main__":
    _generate_example_data()
