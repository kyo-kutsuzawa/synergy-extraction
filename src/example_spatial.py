import numpy as np
import matplotlib.pyplot as plt
from spatial import SpatialSynergy


def example():
    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  6  # Number of DoF
    K =  2  # Number of synergies

    # Create a dataset with shape (N, T, M)
    data, synergies, _ = generate_data(N, T, M, K, noise=0.1)
    print("Data shape    :", data.shape)

    # Extract synergies
    model = SpatialSynergy(K, method="pca")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct time series
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Create a figure
    fig = plt.figure(figsize=(6, 6))

    # Plot reconstruction data
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title("Original/Reconstruction data")
    M_col = np.ceil(np.sqrt(M))
    M_row = np.ceil(M/M_col)
    axes = [fig.add_subplot(M_row*2, M_col, m+1) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color="C{}".format(n))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m],   lw=1, color="C{}".format(n))
            ax.set_xlim((0, data.shape[1]-1))
            ax.set_ylabel("DoF #{}".format(m+1))

    # Plot synergy components
    ax = fig.add_subplot(2, 1, 2)
    ax.set_title("Synergy components")
    for k in range(K):
        x = np.linspace(-0.5, 0.5, M+2)[1:-1] + k
        ax.bar(x, model.synergies[k, :], width=0.95/(M+1), linewidth=0, align='center', color="C{}".format(k))
        ax.bar(x, synergies[k, :],       width=0.25/(M+1), linewidth=0, align='center', color="gray")
    ax.set_xticks(list(range(K)))
    ax.set_xticklabels(["synergy #{}".format(k+1) for k in range(K)])

    #fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.tight_layout()
    plt.show()


def generate_data(N, T, M, K, noise=0.0):
    synergies = np.random.uniform(-1, 1, (K, M))
    activities = np.cumsum(np.random.normal(0, 1.0, (N, T, K)), axis=1)
    data = np.einsum("km,ntk->ntm", synergies, activities)

    data += np.random.normal(0, noise, size=data.shape)  # Add Gaussian noise

    return data, synergies, activities


if __name__ == "__main__":
    example()
