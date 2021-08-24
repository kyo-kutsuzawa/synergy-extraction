import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import timevarying


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract time-varying synergies from a dataset.")
    parser.add_argument("--dirname", type=str, default="data/dataset1/")
    parser.add_argument("--n_synergies", "-K", type=int, default=3, help="Number of synergies in a repertory")
    parser.add_argument("--n_synergy_use", "-D", type=int, default=10, help="Number of synergies used in a data")
    parser.add_argument("--synergy_length", "-S", type=int, default=120, help="Time length of synergies")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument('--noplot', action='store_false', dest="plot")
    parser.add_argument("--out", default="result", help="Output directory.")
    args = parser.parse_args()

    # Load dataset
    data = load_dataset(args.dirname)

    # Setup constants
    N = data.shape[0]  # Number of data
    M = data.shape[2]  # Number of DoF
    K = args.n_synergies
    D = args.n_synergy_use
    S = args.synergy_length

    # Get synergies
    model = timevarying.TimeVaryingSynergy(K, D, S, containing_negative_values=True, mu_w=1e-3, mu_c=1e-3)
    model.extract(data, max_iter=args.n_iter)

    # Reconstruct actions
    data_est = np.empty_like(data)
    activities = model.encode(data, max_iter=2000, mu_c=1e-3)
    data_est = model.decode(activities)

    # Save extracted synergies and activities
    save_result(data, model.synergies, activities, args.out)

    if args.plot:
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
                ax.plot(np.arange(model.synergies.shape[1]), +model.synergies[k, :, m],   lw=1, color="C{}".format(m))
                ax.plot(np.arange(model.synergies.shape[1]), -model.synergies[k, :, m+M], lw=1, color="C{}".format(m))
            ax.set_xlim((0, model.synergies.shape[1]-1))

        plt.show()


def load_dataset(dirname):
    import os, glob

    dataset = []

    # Load trajectory dataset
    filelist = glob.glob(os.path.join(dirname, "*.csv"))
    for filename in filelist:
        data = np.loadtxt(filename, delimiter=",")
        trajectory = data[:, 1:]
        dataset.append(trajectory)

    # Get maximum time length in the dataset
    length = max([d.shape[0] for d in dataset])

    # Padding process
    for i in range(len(dataset)):
        if dataset[i].shape[0] != length:
            d = dataset[i]
            l = d.shape[0]
            dataset[i] = np.full((length, d.shape[1]), d[-1, :])
            dataset[i][0:l, :] = d

    # Concatenate them into a single array
    trajectories = np.stack(dataset, axis=0)

    return trajectories


def save_result(data, synergies, activities, out_dir):
    import os

    amplitude, delays = activities
    N = len(amplitude)
    T = data.shape[1]
    M = data.shape[2]
    K = synergies.shape[0]

    # Save synergies
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "synergy.npy"), synergies)

    # Save data and activities
    for n in range(N):
        # Setup result data
        result = np.zeros((T, M + K))
        result[:, 0:M] = data[n, :, :]

        # Convert the activities to time series
        for k in range(K):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                result[ts, M + k] = c

        # Save the result
        filename = os.path.join(out_dir, "data{}.csv".format(n))
        np.savetxt(filename, result, delimiter=",")


if __name__ == "__main__":
    main()
