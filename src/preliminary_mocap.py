import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import timevarying


def main():
    trajectories = generate_dataset()
    print(trajectories.shape)

    times = np.arange(trajectories.shape[1])

    # Extact motor synergies
    extract_tv(trajectories, times)


def extract_tv(data, times):
    # Setup constants
    N =  15  # Number of data
    M =   3  # Number of DoF
    K =   3  # Number of synergies
    S =  60  # Time length of synergies
    n_iter = 2000

    fig = plt.figure(constrained_layout=True)
    for m in range(M):
        ax = fig.add_subplot(3, 1, m+1)
        for n in range(N):
            ax.plot(np.arange(data.shape[1]), data[n, :, m])
    plt.show()

    # Get synergies
    model = timevarying.TimeVaryingSynergy(K, S, containing_negative_values=True, mu_w=5e-3, mu_c=5e-3)
    model.extract(data, max_iter=n_iter)

    # Reconstruct actions
    data_est = np.empty_like(data)
    activities = model.encode(data, max_iter=1000, mu_c=5e-3)
    data_est = model.decode(activities)
    print(activities)

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


def generate_dataset():
    dataset = []
    for i in range(15):
        theta = np.pi / 4 * (i%5)
        x = np.array([np.cos(theta), 0.0, np.sin(theta)]) * 0.2
        data = reaching(x, 60)

        idx = np.random.randint(0, 40)
        trajectory = np.zeros((100, 3))
        trajectory[idx:idx+60, :] = data
        dataset.append(trajectory)

    trajectories = np.stack(dataset, axis=0)

    return trajectories


def multidirectional(N):
    trajectory = []
    for i in range(5):
        theta = np.pi / 4 * (i%5)
        x_goal = np.array([-np.cos(theta), 0.0, -np.sin(theta)]) * 0.2
        x = reaching(x_goal, N)
        trajectory.append(x)

    trajectory = np.concatenate(trajectory)
    trajectory = np.expand_dims(trajectory, axis=0)
    return trajectory


def reaching(x_goal, N):
    N1 = int(N / 2)
    N2 = N - N1
    x0 = np.zeros(3)
    X1 = s_curve(x0, x_goal, N1)
    X2 = s_curve(x_goal, x0, N2)
    X = np.concatenate([X1, X2], axis=0)

    N3 = int(N / 4)
    x1 = np.array([0, 0.01, 0])
    X3 = s_curve(x0, x1, N3)
    X4 = s_curve(x1, x0, N3)
    X += np.concatenate([X3, X4, X3, X4], axis=0)

    return X


def s_curve(x0, x_goal, N):
    x = np.zeros((N, 3))
    v = np.zeros((N, 3))
    a = np.zeros((N, 3))
    A = 32 / N**3 * (x_goal - x0)

    for n in np.arange(N):
        if n == 0:
            x[n, :] = x0.copy()
            continue

        if n < N * 0.25:
            u = A * n
        elif n < N * 0.75:
            u = A * N * 0.25 - A * (n - N * 0.25)
        elif n < N:
            u = -A * N * 0.25 + A * (n - N * 0.75)

        a[n, :] = u
        x[n, :] = x[n-1, :] + v[n-1, :] + u * 0.5
        v[n, :] = v[n-1, :] + u

    return x


def example_reaching():
    T = 2.0
    N = 1000
    times = np.linspace(0, T, N, endpoint=False)
    x0 = np.zeros(3)
    x_goal = np.array([0.1, 0.0, 0.3])
    x = reaching(x_goal, N)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    ax1.plot(x[:, 0], x[:, 1], x[:, 2])
    ax2.plot(times, x)

    plt.show()


def example_multidirectional():
    # Setup constants
    N = 1  # Number of data
    M = 3  # Number of DoF

    trajectories = multidirectional(100)
    print(trajectories.shape)

    fig = plt.figure(constrained_layout=True)
    for m in range(M):
        ax = fig.add_subplot(3, 1, m+1)
        for n in range(N):
            ax.plot(np.arange(trajectories.shape[1]), trajectories[n, :, m])
    plt.show()


if __name__ == "__main__":
    example_multidirectional()
    #main()
