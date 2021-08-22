import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from timevarying import TimeVaryingSynergy
from timevarying import update_delays, update_amplitude, update_synergies


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", nargs="+", choices=["all", "entire", "delay", "amplitude", "synergy", "data"], default="entire")
    args = parser.parse_args()

    if "data" in args.task or "all" in args.task:
        generate_example_data(plot=True)

    if "delay" in args.task or "all" in args.task:
        example_update_delays()

    if "amplitude" in args.task or "all" in args.task:
        example_update_amplitude()

    if "synergy" in args.task or "all" in args.task:
        example_update_synergies()

    if "entire" in args.task or "all" in args.task:
        example()


def example():
    """Check synergy extraction code.
    """
    import matplotlib.pyplot as plt

    # Setup constants
    N =  10  # Number of data
    M =   3  # Number of DoF
    T = 150  # Time length of data
    K =   2  # Number of synergies
    S =  50  # Time length of synergies
    max_iter = 2000

    # Create a dataset with shape (N, T, M)
    data, synergies, (amplitude, delays) = generate_example_data(N, M, T, K, S, plot=False)
    #data += abs(np.random.normal(0, 0.1, size=data.shape))  # Add Gaussian noise

    # Get synergies
    model = TimeVaryingSynergy(K, S)
    model.extract(data, max_iter=max_iter)

    # Reconstruct actions
    data_est = np.empty_like(data)
    activities = model.encode(data, max_iter=max_iter)
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
            ax.plot(np.arange(synergies.shape[1]), synergies[k, :, m], "--", lw=2, color="C{}".format(m))
            ax.plot(np.arange(model.synergies.shape[1]), model.synergies[k, :, m], lw=1, color="C{}".format(m))
        ax.set_xlim((0, model.synergies.shape[1]-1))

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

    # Generate synergy activities (i.e., amplitude and delay)
    refractory_period = np.ceil(S * 0.5).astype(np.int)

    # Determine the numbers of synergies to be used
    synergy_use = np.random.uniform(0, 1, (N, K))
    synergy_use = synergy_use / np.sum(synergy_use, axis=1, keepdims=True) * D
    synergy_use = np.round(synergy_use).astype(np.int)
    synergy_use[:, -1] = D - np.sum(synergy_use[:, :-1], axis=1)

    # Compute delays
    delays = []
    for n in range(N):
        delays.append([])
        for k in range(K):
            delays[n].append([])
            margin = T - S * synergy_use[n, k] - refractory_period * (synergy_use[n, k] - 1)

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
    data = np.zeros((N, T, M))
    for n in range(N):
        for k in range(K):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data[n, ts:ts+S, :] += c * synergies[k, :, :]

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


def example_update_delays():
    import matplotlib.pyplot as plt

    # Setup constants
    N =  10  # Number of data
    M =   3  # Number of DoF
    T = 150  # Time length of data
    K =   2  # Number of synergies
    S =  50  # Time length of synergies

    # Create a dataset with shape (N, T, M)
    data, synergies, (amplitude, delays) = generate_example_data(N, M, T, K, S, plot=False)

    # Estimate delays
    delays_est = update_delays(data, synergies)

    # Print the results
    print("Actual:\n", delays)
    print("Expect:\n", delays_est)
    print("Residual:\n", delays - delays_est)

    # Reconstruct the data
    data_est = np.zeros_like(data)
    for n in range(N):
        for k in range(K):
            ts = delays_est[n, k]
            data_est[n, ts:ts+S, :] += amplitude[n, k] * synergies[k, :, :]

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
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", color="C{}".format(n))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m], color="C{}".format(n))
            ax.set_xlim((0, data.shape[1]-1))

    # Plot synergy components
    gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
    for k in range(K):
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(synergies.shape[1]), synergies[k, :, m], color="C{}".format(m))
        ax.set_xlim((0, synergies.shape[1]-1))

    plt.show()


def example_update_amplitude():
    import matplotlib.pyplot as plt

    # Setup constants
    N =  10  # Number of data
    M =   3  # Number of DoF
    T = 150  # Time length of data
    K =   2  # Number of synergies
    S =  50  # Time length of synergies
    n_iter = 1000
    mu = 1e-4

    # Create a dataset with shape (N, T, M)
    data, synergies, (amplitude, delays) = generate_example_data(N, M, T, K, S, plot=False)

    # Estimate amplitude
    amplitude_est = np.random.uniform(0, 1, amplitude.shape)
    for _ in range(n_iter):
        amplitude_est = update_amplitude(data, synergies, amplitude_est, delays, mu)

    # Print the results
    print("Actual:\n", amplitude)
    print("Expect:\n", amplitude_est)
    print("Residual:\n", amplitude - amplitude_est)

    # Reconstruct the data
    data_est = np.zeros_like(data)
    for n in range(N):
        for k in range(K):
            ts = delays[n, k]
            data_est[n, ts:ts+S, :] += amplitude_est[n, k] * synergies[k, :, :]

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
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", color="C{}".format(n))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m], color="C{}".format(n))
            ax.set_xlim((0, data.shape[1]-1))

    # Plot synergy components
    gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
    for k in range(K):
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(synergies.shape[1]), synergies[k, :, m], "--", color="C{}".format(m))
        ax.set_xlim((0, synergies.shape[1]-1))

    plt.show()


def example_update_synergies():
    import matplotlib.pyplot as plt

    # Setup constants
    # Setup constants
    N =  10  # Number of data
    M =   3  # Number of DoF
    T = 150  # Time length of data
    K =   2  # Number of synergies
    S =  50  # Time length of synergies
    n_iter = 1000
    mu = 1e-3

    # Create a dataset with shape (N, T, M)
    data, synergies, (amplitude, delays) = generate_example_data(N, M, T, K, S, plot=False)

    # Estimate synergies
    synergies_est = np.random.uniform(0, 1, synergies.shape)
    for _ in range(n_iter):
        synergies_est = update_synergies(data, synergies_est, amplitude, delays, mu)

    # Print the results
    print("Actual:\n", synergies)
    print("Expect:\n", synergies_est)
    print("Residual:\n", synergies - synergies_est)

    # Reconstruct the data
    data_est = np.zeros_like(data)
    for n in range(N):
        for k in range(K):
            ts = delays[n, k]
            data_est[n, ts:ts+S, :] += amplitude[n, k] * synergies_est[k, :, :]

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
            ax.plot(np.arange(data.shape[1]), data[n, :, m], "--", color="C{}".format(n))
            ax.plot(np.arange(data.shape[1]), data_est[n, :, m], color="C{}".format(n))
            ax.set_xlim((0, data.shape[1]-1))

    # Plot synergy components
    gs_2 = GridSpecFromSubplotSpec(nrows=K, ncols=1, subplot_spec=gs_master[0, 1])
    for k in range(K):
        ax = fig.add_subplot(gs_2[k, :])
        ax.set_title("synergy #{}".format(k+1))
        for m in range(M):
            ax.plot(np.arange(synergies.shape[1]), synergies[k, :, m], "--", color="C{}".format(m))
            ax.plot(np.arange(synergies_est.shape[1]), synergies_est[k, :, m], color="C{}".format(m))
        ax.set_xlim((0, synergies.shape[1]-1))

    plt.show()


if __name__ == "__main__":
    main()
