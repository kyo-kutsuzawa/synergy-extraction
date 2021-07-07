import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from spatiotemporal import SpatioTemporalSynergy


def example():
    import matplotlib.pyplot as plt

    # Setup constants
    N =  5  # Number of data
    T = 20  # Time length
    M =  4  # Number of DoF
    K =  3  # Number of synergies

    # Create a dataset with shape (N, T, M)
    data = generate_data(N, T, M, K, noise=0.2)
    print("Data shape    :", data.shape)

    # Get synergies
    model = SpatioTemporalSynergy(K, method="pca")
    model.extract(data)
    print("Synergy shape :", model.synergies.shape)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)

    # Create a figure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs_master = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[2, 1])

    # Plot reconstruction data
    M_col = int(np.ceil(np.sqrt(M)))
    M_row = int(np.ceil(M/M_col))
    gs_1 = GridSpecFromSubplotSpec(nrows=M_row, ncols=M_col, subplot_spec=gs_master[0, 0])
    axes = [fig.add_subplot(gs_1[int(np.floor(m/M_col)), m%M_col]) for m in range(M)]
    for n in range(N):
        for m, ax in enumerate(axes):
            ax.set_title("DoF #{}".format(m+1))
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color="C{}".format(n))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m],   lw=1, color="C{}".format(n))
            ax.set_xlim((0, data.shape[1]-1))

    # Plot synergy components
    gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
    for k in range(K):
        #ax = fig.add_subplot(2*K, 1, K+k+1)
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(model.synergies.shape[1]), model.synergies[k, :, m], color="C{}".format(m))
        ax.set_xlim((0, model.synergies.shape[1]-1))

    plt.show()


def generate_data(N, T, M, K, noise=0.0):
    synergies = np.cumsum(np.random.normal(0, 1, (K, T, M)), axis=1)
    activities = np.random.uniform(-1, 1, (N, K))
    data = np.einsum("ktm,nk->ntm", synergies, activities)

    data += np.random.normal(0, noise, size=data.shape)  # Add Gaussian noise

    return data


if __name__ == "__main__":
    example()
