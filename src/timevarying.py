import numpy as np


class TimeVaryingSynergy:
    """Time-varying synergies.
    """

    def __init__(self, n_synergies, synergy_length, containing_negative_values=False, mu_w=1e-3, mu_c=1e-3):
        """
        Args:
            n_synergies: Number of synergies
        """
        self.n_synergies = n_synergies
        self.synergy_length = synergy_length
        self.containing_negative_values = containing_negative_values

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.data_length = None
        self.mu_w = mu_w
        self.mu_c = mu_c

    def extract(self, data, max_iter=10000):
        """Extract time-varying synergies from given data.

        Data is assumed to have the shape (#data, data-length, #DoF).
        Synergies have the shape (#synergies, synergy-length, #DoF).
        """
        # Get shape information
        self.dof = data.shape[2]
        self.data_length = data.shape[1]

        # Convert the data to non-negative signals
        if self.containing_negative_values:
            data = transform_nonnegative(data)
            self.dof = data.shape[2]  # Update the number of DoF

        # Initialize synergies
        data_mean = np.mean(data)
        self.synergies = np.random.uniform(0.0, data_mean*2, (self.n_synergies, self.synergy_length, self.dof))
        amplitude      = np.random.uniform(0.01, 1.0, (data.shape[0], self.n_synergies))

        # Extraction loop
        for i in range(max_iter):
            delays = update_delays(data, self.synergies)
            amplitude = update_amplitude(data, self.synergies, amplitude, delays, self.mu_c)
            self.synergies = update_synergies(data, self.synergies, amplitude, delays, self.mu_w)

            if i % 100 == 0:
                r2 = compute_R2(data, self.synergies, amplitude, delays)
                print("Iter {:4d}: R2 = {}".format(i, r2))

        return self.synergies

    def encode(self, data, max_iter=1000, mu_c=None):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, data-length, #DoF).
        Synergy activities are represented a tuple of the amplitude and delay; both have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Convert the data to non-negative signals
        if self.containing_negative_values:
            data = transform_nonnegative(data)

        if mu_c is None:
            mu_c = self.mu_c

        # Encoding loop
        amplitude = np.random.uniform(0.01, 1.0, size=(data.shape[0], self.n_synergies))
        for i in range(max_iter):
            delays = update_delays(data, self.synergies)
            amplitude = update_amplitude(data, self.synergies, amplitude, delays, mu_c)

            print("Encoding... {}% ({}/{})".format(int(i/max_iter*100), i, max_iter), end="\r")

        activities = (amplitude, delays)

        return activities

    def decode(self, activities):
        """Decode given synergy activities to data.

        Synergy activities are assumed to have the shape (#trajectories, #activities).
        Data have the shape (#trajectories, data-length, #DoF).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Reconstruct data
        amplitude, delays = activities
        data = np.zeros((activities[0].shape[0], self.data_length, self.dof))
        for n in range(data.shape[0]):
            for k in range(self.n_synergies):
                c = amplitude[n, k]
                ts = delays[n, k]
                data[n, ts:ts+self.synergy_length, :] += c * self.synergies[k, :, :]

        # Convert non-negative signals backwards
        if self.containing_negative_values:
            data = inverse_transform_nonnegative(data)

        return data


def match_synergies(data, synergies, n_synergy_use, refractory_period):
    """Find the delays and amplitude with matching pursuit.

    The algorithm is based on [d'Avella et al., 2005].
    """
    # Setup variables
    data = data.copy()
    n_data = data.shape[0]
    data_length = data.shape[1]
    synergy_length = synergies.shape[1]
    n_synergies = synergies.shape[0]

    # Initialize delays
    delays = [[[] for _ in range(n_synergies)] for _ in range(data.shape[0])]
    amplitude = [[[] for _ in range(n_synergies)] for _ in range(data.shape[0])]

    # Find delay times for each data sequence
    for n in range(n_data):
        synergy_available = np.full((n_synergies, data_length), True)  # Whether the delay time of the synergy has been found
        for d in range(n_synergy_use):
            # Compute dot products for all possible patterns
            corr = np.zeros((n_synergies, data_length))  # Whether the delay time of the synergy has been found
            for k in range(n_synergies):
                for ts in range(data_length - synergy_length):
                    if synergy_available[k, ts]:
                        corr[k, ts] = np.sum(data[n, ts:ts+synergy_length, :] * synergies[k])

            # Register maximum value
            k, ts = np.unravel_index(np.argmax(corr), corr.shape)
            c = np.max(corr)
            delays[n][k].append(ts)

            # Subtract the selected pattern
            data[n, ts:ts+synergy_length, :] -= c * synergies[k]

            # Remove the selected pattern and its surroundings
            t0 = max(ts - refractory_period, 0)
            t1 = min(ts + refractory_period, data_length)
            synergy_available[k, t0:t1] = False

        for k in range(n_synergies):
            delays[n][k].sort()

    return delays


def update_delays(data, synergies):
    """Find the delays using a nested matching procedure based on the cross-correlation.

    The algorithm is based on [d'Avella et al., 2002].
    """
    # Setup variables
    data = data.copy()
    data_length = data.shape[1]
    synergy_length = synergies.shape[1]
    n_synergies = synergies.shape[0]

    # Initialize delays
    delays = np.zeros((data.shape[0], n_synergies), dtype=np.int)

    # Find delay times for each data sequence
    for n in range(data.shape[0]):
        synergy_substracted = np.full((n_synergies,), False)  # Whether the delay time of the synergy has been found
        for k in range(n_synergies):
            # Compute correlation functions for each synergies
            corrs = []
            for k in range(n_synergies):
                # Initialize the cross-correlation between the n-th data and k-th synergy.
                # Note that the minimum possible value is zero;
                # Default values are -1 so as to the previously-selected synergy is never selected again.
                corr = -np.ones((data_length-synergy_length+1,))

                # Compute the cross-correlation if its delay has not been found yet
                if not synergy_substracted[k]:
                    for m in range(data.shape[2]):
                        corr += np.correlate(data[n, :, m], synergies[k, :, m])
                corrs.append(corr)

            # Select the synergy and delay with highest cross-correlation
            corrs = np.array(corrs)
            (idx_synergy, delay) = np.unravel_index(np.argmax(corrs), corrs.shape)
            delays[n, idx_synergy] = delay

            # Substract the scaled and shifted synergy from the data
            coef = corrs[idx_synergy, delay] / synergy_length
            data[n, delay:delay+synergy_length, :] -= coef * synergies[idx_synergy, :, :]
            synergy_substracted[idx_synergy] = True

    return delays


def update_amplitude(data, synergies, amplitude, delays, mu=0.001):
    """Find the amplitude (scale coefficient).

    The algorithm is based on [d'Avella and Tresch, 2002].
    """
    # Compute shifted synergies and reconstruction data
    shifted_synergies = np.zeros((data.shape[0], data.shape[1], data.shape[2], synergies.shape[0]))
    data_est = np.zeros_like(data)
    for n in range(data.shape[0]):
        for k in range(synergies.shape[0]):
            ts = delays[n, k]
            shifted_synergies[n, ts:ts+synergies.shape[1], :, k] += synergies[k, :, :]
            data_est[n, ts:ts+synergies.shape[1], :] += synergies[k, :, :] * amplitude[n, k]

    # Compute the gradient
    grad = -2 * np.einsum("ntm,ntmk->nk", data - data_est, shifted_synergies)

    # Update the amplitude
    amplitude = amplitude - mu * grad
    amplitude = np.clip(amplitude, 0.0, None)  # Limit to non-negative values

    return amplitude


def update_synergies(data, synergies, amplitude, delays, mu=0.001):
    """Find synergies.

    The algorithm is based on [d'Avella and Tresch, 2002].
    """
    # Compute reconstruction data
    data_est = np.zeros_like(data)
    for n in range(data.shape[0]):
        for k in range(synergies.shape[0]):
            ts = delays[n, k]
            data_est[n, ts:ts+synergies.shape[1], :] += synergies[k, :, :] * amplitude[n, k]

    # Compute the gradient
    deviation = data - data_est
    grad = np.zeros_like(synergies)
    for k in range(synergies.shape[0]):
        for n in range(data.shape[0]):
            ts = delays[n, k]
            grad[k, :, :] += deviation[n, ts:ts+synergies.shape[1], :] * amplitude[n, k]

    # Compute the gradient
    grad = grad * -2

    # Update the amplitude
    synergies = synergies - mu * grad
    synergies = np.clip(synergies, 0.0, None)  # Limit to non-negative values

    return synergies


def compute_R2(data, synergies, amplitude, delays):
    # Compute reconstruction data
    data_est = np.zeros_like(data)
    for n in range(data.shape[0]):
        for k in range(synergies.shape[0]):
            ts = delays[n, k]
            data_est[n, ts:ts+synergies.shape[1], :] += synergies[k, :, :] * amplitude[n, k]

    # Compute the R2 value
    data_mean = np.mean(data)
    R2 = 1 - np.sum(np.square(data - data_est)) / np.sum(np.square(data - data_mean))

    return R2


def transform_nonnegative(data):
    """Convert a data that has negative values to non-negative signals with doubled dimensions.
    Data is assumed to have the shape (#trajectories, length, #DoF).
    Converted non-negative data have the shape (#trajectories, length, 2 * #DoF).
    """
    n_dof = data.shape[2]  # Dimensionality of the original data

    # Convert the data to non-negative signals
    data_nn = np.empty((data.shape[0], data.shape[1], n_dof*2))
    data_nn[:, :, :n_dof] = +np.maximum(data, 0.0)
    data_nn[:, :, n_dof:] = -np.minimum(data, 0.0)

    return data_nn


def inverse_transform_nonnegative(data):
    """Inverse conversion of `transform_nonnegative()`; Convert non-negative signals to a data that has negative values.
    Non-negative data is assumed to have the shape (#trajectories, length, 2 * #DoF).
    Reconstructed data have the shape (#trajectories, length, #DoF).
    """
    n_dof = int(data.shape[2] / 2)  # Dimensionality of the original data

    # Restore the original data
    data_rc = np.empty((data.shape[0], data.shape[1], n_dof))
    data_rc = data[:, :, :n_dof] - data[:, :, n_dof:]

    return data_rc
