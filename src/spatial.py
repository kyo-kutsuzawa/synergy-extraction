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


def _example_spatial():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  6  # Number of DoF
    K =  2  # Number of synergies

    # Create a dataset with shape (N, T, M)
    synergies = np.random.uniform(-1, 1, (M, K))
    activities = np.cumsum(np.random.normal(0, 1.0, (N, T, K)), axis=1)
    data = np.einsum("mk,ntk->ntm", synergies, activities)
    data += np.random.normal(0, 0.1, size=data.shape)  # Add Gaussian noise
    print("Data shape    :", data.shape)

    # Get synergies
    model = SpatialSynergy(K, method="pca")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    ax1 = fig.add_subplot(1, 1, 1)
    for k in range(K):
        x = np.linspace(-0.5, 0.5, M+2)[1:-1] + k
        ax1.bar(x, model.synergies[k, :], width=0.95/(M+1), linewidth=0, align='center')
    ax1.set_xticks(list(range(K)))
    ax1.set_xticklabels(["synergy #{}".format(k+1) for k in range(K)])
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
    _example_spatial()
