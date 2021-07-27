import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import spatial, timevarying


def main():
    # Load an EMG data
    filename = "data/data_1e-2.csv"
    data = np.loadtxt(filename, delimiter=",")

    times = data[:, 0]
    positions = data[:, 1:]

    positions = positions.reshape(1, positions.shape[0], positions.shape[1])

    # Extact motor synergies
    #extract_spatial(positions, times)
    extract_tv(positions, times)


def extract_spatial(data, times):
    # Setup a mucle-synergy model
    n_dof = 3
    n_synergies = 2
    #model = spatial.SpatialSynergy(n_synergies, method="pca")
    model = spatial.SpatialSynergy(n_synergies, method="negative-nmf")
    model.extract(data)

    # Reconstruct actions
    activities = model.encode(data)
    data_est = model.decode(activities)
    print("Activity shape:", activities.shape)
    print(model.synergies.shape)

    # Plot synergy components
    fig = plt.figure()
    fig.suptitle("Synergy components")
    ax1 = fig.add_subplot(1, 1, 1)
    for k in range(n_synergies):
        n = model.synergies.shape[1]
        x = np.linspace(-0.5, 0.5, n+2)[1:-1] + k
        ax1.bar(x, model.synergies[k, :], width=0.95/(n_dof+1), linewidth=0, align='center')
    ax1.set_xticks(list(range(n_synergies)))
    ax1.set_xticklabels(["synergy #{}".format(k+1) for k in range(n_synergies)])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Plot reconstruction data
    fig = plt.figure()
    fig.suptitle("Original/Reconstruction data")
    M_row = np.ceil(np.sqrt(n_dof))
    M_col = np.ceil(n_dof/M_row)
    axes = [fig.add_subplot(M_row, M_col, m+1) for m in range(n_dof)]
    for m, ax in enumerate(axes):
        ax.plot(times, data[0, :, m], "--", lw=2)
        ax.plot(times, data_est[0, :, m],      lw=1)
        ax.set_xlim((times[0], times[-1]))
        ax.set_ylabel("DoF #{}".format(m+1))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def extract_tv(data, times):
    # Setup constants
    N =   1  # Number of data
    M =   3  # Number of DoF
    K =   3  # Number of synergies
    S = 300  # Time length of synergies

    #fig = plt.figure()
    #for m in range(M):
    #    ax = fig.add_subplot(4, 1, m+1)
    #    ax.plot(np.arange(data.shape[1]), data[0, :, m])

    # Get synergies
    model = timevarying.TimeVaryingSynergy(K, S, containing_negative_values=True)
    model.extract(data, max_iter=10000)

    # Reconstruct actions
    data_est = np.empty_like(data)
    activities = model.encode(data)
    data_est = model.decode(activities)

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
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(model.synergies.shape[1]), model.synergies[k, :, m], lw=1, color="C{}".format(m))
        ax.set_xlim((0, model.synergies.shape[1]-1))

    plt.show()


if __name__ == "__main__":
    main()
