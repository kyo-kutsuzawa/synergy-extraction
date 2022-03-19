import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import timevarying


def example():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  10  # Number of data
    M =   3  # Number of DoF
    T = 150  # Time length of data
    K =   2  # Number of synergies in a repertory
    D =   4  # Number of synergies used in a data
    S =  20  # Time length of synergies
    n_iter = 100
    lr = 0.005
    refractory_period = int(S * 0.5)

    # Create a dataset with shape (N, T, M)
    dataset, synergies, (amplitude, delays) = generate_example_data(N, M, T, K, D, S, plot=False)

    # Extract motor synergies
    synergies_est, delays_est, amplitude_est, dataset_rec = timevarying.extract(dataset, n_synergies=K, synergy_length=S, n_dof=M, n_iter=n_iter, lr=lr, refractory_period=refractory_period)

    # Create a figure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs_master = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[2, 1])

    # Plot reconstruction data
    gs_1 = GridSpecFromSubplotSpec(nrows=M, ncols=1, subplot_spec=gs_master[0, 0])
    axes = [fig.add_subplot(gs_1[m, :]) for m in range(M)]
    for n in range(N):
        data = dataset[n]
        data_rec = dataset_rec[n]

        for m, ax in enumerate(axes):
            ax.set_title("DoF #{}".format(m+1))
            ax.plot(np.arange(data.shape[0]), data[:, m], "--", lw=2, color="C{}".format(n))
            ax.plot(np.arange(data.shape[0]), data_rec[:, m],   lw=1, color="C{}".format(n))
            ax.set_xlim((0, data.shape[0] - 1))

    # Plot synergy components
    gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
    for k in range(K):
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(synergies.shape[1]),     synergies[k, :, m], "--", lw=2, color="C{}".format(m))
            ax.plot(np.arange(synergies_est.shape[1]), synergies_est[k, :, m], lw=1, color="C{}".format(m))
        ax.set_xlim((0, synergies.shape[1]-1))

    plt.show()


def generate_example_data(N=3, M=3, T=100, K=3, D=4, S=15, plot=True):
    """Check example-data generation code.

    N: Number of data
    M: Number of DoF
    T: Time length of data
    K: Number of synergies in a repertory
    D: Number of synergies used in a data
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

    for k in range(synergies.shape[0]):
        norm = np.sqrt(np.sum(np.square(synergies[k])))
        synergies[k] = synergies[k] / float(norm)

    # Generate synergy activities (i.e., amplitude and delay)
    refractory_period = np.ceil(S * 0.5).astype(np.int)

    # Determine the numbers of synergies to be used
    synergy_use = np.random.uniform(0, 1, (N, K))
    synergy_use = synergy_use / np.sum(synergy_use, axis=1, keepdims=True) * D
    synergy_use = np.round(synergy_use).astype(np.int)
    synergy_use[:, -1] = D - np.sum(synergy_use[:, :-1], axis=1)

    # Generate a dataset
    lengths = [int(T * np.random.uniform(0.8, 1.5)) for n in range(N)]

    # Compute delays
    delays = []
    for n in range(N):
        delays.append([])
        for k in range(K):
            delays[n].append([])
            margin = lengths[n] - S * synergy_use[n, k] - refractory_period * (synergy_use[n, k] - 1)

            if margin < 0:
                return

            ts = 0
            for l in range(synergy_use[n, k]):
                delta_ts = np.random.randint(margin)
                ts += delta_ts
                margin -= delta_ts
                if l >= 1:
                    ts += refractory_period
                delays[n][k].append(ts)
                ts += S

    # Compute amplitude
    amplitude = []
    for delays_n in delays:
        amplitude.append([])
        for delays_nk in delays_n:
            amplitude[-1].append([])
            for _ in delays_nk:
                c = np.random.uniform(0, 1)
                amplitude[-1][-1].append(c)

    # Compute a dataset from the synergies and activities
    dataset = []
    for n in range(N):
        data = np.zeros((lengths[n], M))
        for k in range(K):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data[ts:ts+S, :] += c * synergies[k, :, :]

        data += abs(np.random.normal(0, 0.01, size=data.shape))  # Add Gaussian noise

        dataset.append(data)

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
            data = dataset[n]
            for m, ax in enumerate(axes):
                ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", lw=2, color=plt.get_cmap("viridis")((N-n)/(N+1)))
                ax.set_xlim((0, data.shape[1]-1))
                ax.set_ylabel("DoF #{}".format(m+1))
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plt.show()

    return dataset, synergies, (amplitude, delays)


if __name__ == "__main__":
    example()
