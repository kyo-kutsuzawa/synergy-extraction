import numpy as np


def match_synergies(dataset, synergies, n_synergy_use, refractory_period, amplitude_threshold):
    """
    Find the delays and amplitude with matching pursuit.

    The algorithm is based on [d'Avella et al., 2005].
    """
    # Setup variables
    n_data = len(dataset)
    synergy_length = synergies.shape[1]
    n_synergies = synergies.shape[0]

    # Initialize delays
    delays = [[[] for _ in range(n_synergies)] for _ in range(n_data)]
    amplitude = [[[] for _ in range(n_synergies)] for _ in range(n_data)]

    # Find delay times for each data sequence
    for n in range(n_data):
        residual = dataset[n].copy()
        data_length = data.shape[0]

        synergy_available = np.full((n_synergies, data_length), True)  # Whether the delay time of the synergy can be used
        for d in range(n_synergy_use):
            # Compute dot products for all possible patterns
            corr = np.zeros((n_synergies, data_length))
            for k in range(n_synergies):
                for ts in range(data_length - synergy_length):
                    if synergy_available[k, ts]:
                        corr[k, ts] = np.sum(residual[ts:ts+synergy_length, :] * synergies[k])

            # Register the best-matching pattern
            k, ts = np.unravel_index(np.argmax(corr), corr.shape)
            c = np.max(corr) / np.sum(synergies[k] ** 2)

            # Finish matching when the correlation is lower than the threshold
            if c < amplitude_threshold:
                break

            delays[n][k].append(ts)
            amplitude[n][k].append(c)

            # Subtract the selected pattern
            residual[ts:ts+synergy_length, :] -= c * synergies[k]

            # Remove the selected pattern and its surroundings
            t0 = max(ts - refractory_period, 0)
            t1 = min(ts + refractory_period, data_length)
            synergy_available[k, t0:t1] = False

    return delays, amplitude


def update_synergies(dataset, synergies, delays, amplitude, mu=0.001):
    """
    Update synergies with gradient descent.

    The algorithm is based on [d'Avella and Tresch, 2002].
    """
    n_data = len(dataset)
    grad = np.zeros_like(synergies)

    for n in range(n_data):
        data = dataset[n]

        # Compute reconstruction data
        data_est = np.zeros_like(data)
        for k in range(synergies.shape[0]):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]

        # Compute the gradient
        deviation = data - data_est
        for k in range(synergies.shape[0]):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                grad[k, :, :] += deviation[ts:ts+synergies.shape[1], :] * c

    # Compute the gradient
    grad = grad * -2

    # Update the amplitude
    synergies = synergies - mu * grad
    synergies = np.clip(synergies, 0.0, None)  # Limit to non-negative values

    for k in range(synergies.shape[0]):
        norm = np.sqrt(np.sum(np.square(synergies[k])))
        synergies[k] = synergies[k] / float(norm)

    return synergies


def decode(delays, amplitude, synergies, lengths):
    n_data = len(delays)
    n_synergies, synergy_length, dof = synergies.shape

    dataset = []
    for n in range(n_data):
        data_length = lengths[n]

        data = np.zeros((data_length, dof))
        for k in range(n_synergies):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data[ts:ts + synergy_length, :] += c * synergies[k, :, :]

        dataset.append(data)

    return dataset


def compute_R2(dataset, synergies, delays, amplitude):
    n_data = len(dataset)

    mse_sum = 0.0

    for n in range(n_data):
        data = dataset[n]

        # Compute reconstruction data
        data_est = np.zeros_like(data)
        for k in range(synergies.shape[0]):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]

        mse_sum += np.sum(np.square(data - data_est))

    data_cat = np.concatenate(dataset, axis=0)
    data_mean = np.mean(data_cat)

    # Compute the R2 value
    R2 = 1 - mse_sum / np.sum(np.square(data_cat - data_mean))

    return R2


def compute_mse(dataset, synergies, delays, amplitude):
    n_data = len(dataset)

    mse_sum = 0.0

    for n in range(n_data):
        data = dataset[n]

        # Compute reconstruction data
        data_est = np.zeros_like(data)
        for k in range(synergies.shape[0]):
            for ts, c in zip(delays[n][k], amplitude[n][k]):
                data_est[ts:ts+synergies.shape[1], :] += c * synergies[k, :, :]

        mse_sum += np.sum(np.square(data - data_est))

    return mse_sum


def transform_nonnegative(dataset):
    """
    Convert a data that has negative values to non-negative signals with doubled dimensions.
    The dataset is assumed to be a list of trajectories with shape of (length, DoF).
    The result dataset is a list of trajectories with shape of (length, 2 * DoF).
    The first half in DoF-axis corresponds to the positive values, whereas the second half corresponds to the negative values.
    """
    dataset_nn = []

    for data in dataset:
        n_dof = data.shape[1]  # Dimensionality of the original data

        # Convert the data to non-negative signals
        data_nn = np.empty((data.shape[0], n_dof*2))
        data_nn[:, :n_dof] = +np.maximum(data, 0.0)
        data_nn[:, n_dof:] = -np.minimum(data, 0.0)

        dataset_nn.append(data_nn)

    return dataset_nn


def inverse_transform_nonnegative(dataset):
    """
    Inverse conversion of `transform_nonnegative()`; Convert a non-negative dataset to another dataset that contains negative values.
    The non-negative dataset is assumed to be a list of trajectories with shape of (length, DoF).
    The reconstructed dataset is a list of trajectories with shape of (length, DoF / 2).
    """
    dataset_rc = []

    for data in dataset:
        n_dof = int(data.shape[1] / 2)  # Dimensionality of the original data

        # Restore the original data
        data_rc = np.empty((data.shape[0], n_dof))
        data_rc = data[:, :n_dof] - data[:, n_dof:]

        dataset_rc.append(data_rc)

    return dataset_rc
