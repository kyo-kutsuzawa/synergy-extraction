import numpy as np
from sklearn.decomposition import PCA, NMF


class SpatioTemporalSynergy:
    """Spatiotemporal synergies.
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
        """Extract spatiotemporal synergies from given data.

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


def _example_spatiotemporal():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  3  # Number of DoF
    K =  2  # Number of synergies

    # Create a dataset with shape (N, T, M)
    synergies = np.cumsum(np.random.normal(0, 1, (K, T, M)), axis=1)
    activities = np.random.uniform(-1, 1, (N, K))
    data = np.einsum("ktm,nk->ntm", synergies, activities)
    data += np.random.normal(0, 0.1, size=data.shape)  # Add Gaussian noise
    print("Data shape    :", data.shape)

    # Get synergies
    model = SpatioTemporalSynergy(K, method="pca")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    for k in range(K):
        ax = fig.add_subplot(K, 1, k+1)
        for m in range(M):
            ax.plot(np.arange(model.synergies.shape[1]), model.synergies[k, :, m], color=plt.get_cmap("viridis")((M-m)/(M+1)))
        ax.set_xlim((0, model.synergies.shape[1]-1))
        ax.set_ylabel("synergy #{}".format(k+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot reconstruction data
    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data")
    M_row = np.ceil(np.sqrt(M))
    M_col = np.ceil(M/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m],   lw=1, color=plt.get_cmap("viridis")((N-n)/(N+1)))
            ax.set_xlim((0, data.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


if __name__ == "__main__":
    _example_spatiotemporal()
