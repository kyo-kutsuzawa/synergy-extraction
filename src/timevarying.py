import numpy as np
from sklearn.decomposition import NMF


class TimeVaryingSynergy:
    """Time-varying synergies.
    """

    def __init__(self, n_synergies, synergy_length):
        """
        Args:
            n_synergies: Number of synergies
        """
        self.n_synergies = n_synergies
        self.synergy_length = synergy_length

        # Initialize variables
        self.model = None
        self.synergies = None
        self.dof = None
        self.data_length = None

    def extract(self, data, max_iter=10000):
        """Extract time-varying synergies from given data.

        Data is assumed to have the shape (#data, data-length, #DoF).
        Synergies have the shape (#synergies, synergy-length, #DoF).
        """
        # Get shape information
        self.dof = data.shape[2]
        self.data_length = data.shape[1]

        self.synergies = np.random.uniform(0.1, 1, (self.n_synergies, self.synergy_length, self.dof))
        amplitude      = np.random.uniform(0.1, 1, (data.shape[0], self.n_synergies))

        for i in range(max_iter):
            delays = update_delays(data, self.synergies)
            amplitude = update_amplitude(data, self.synergies, amplitude, delays)
            self.synergies = update_synergies(data, self.synergies, amplitude, delays, eps=1e-9)

        return self.synergies

    def encode(self, data, max_iter=1000):
        """Encode given data to synergy activities.

        Data is assumed to have the shape (#trajectories, data-length, #DoF).
        Synergy activities are represented a tuple of the amplitude and delay; both have the shape (#trajectories, #synergies).
        """
        # If synergies have not extracted, throw an exception
        if self.synergies is None:
            return None

        # Not implemented
        amplitude = np.empty((data.shape[0], self.n_synergies))
        delays = np.empty((data.shape[0], self.n_synergies))
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

        return data


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
                # Note that the minimum possible value is zero.
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


def update_amplitude(data, synergies, amplitude, delays):
    """Find the amplitude (scale coefficient).

    The algorithm is based on [d'Avella et al., 2003].
    """
    # Compute shifted synergies (correspond to W Theta[t_s]) and shifted and scaled synergies (correspond to W H_s)
    shifted_synergies = np.zeros_like(data)
    shifted_scaled_synergies = np.zeros_like(data)
    for n in range(data.shape[0]):
        for k in range(synergies.shape[0]):
            ts = delays[n, k]
            shifted_synergies[n, ts:ts+synergies.shape[1], :] += synergies[k, :, :]
            shifted_scaled_synergies[n, ts:ts+synergies.shape[1], :] += synergies[k, :, :] * amplitude[n, k]

    # Update the amplitude
    for n in range(data.shape[0]):
        N = np.dot(data[n].T, shifted_synergies[n])
        D = np.dot(shifted_scaled_synergies[n].T, shifted_synergies[n])
        #amplitude[n, :] = amplitude[n, :] * np.trace(N) / np.trace(D)
        amplitude[n, :] = amplitude[n, :] + 0.001 * (np.trace(N) - np.trace(D))

    return amplitude


def update_synergies(data, synergies, amplitude, delays, eps=1e-9):
    """Find synergies.

    The algorithm is based on [d'Avella et al., 2003].
    Note that the shape setup is different to the original paper due to implementation consistency.
    """
    n_data = data.shape[0]
    data_length = data.shape[1]
    n_synergies = synergies.shape[0]
    synergy_length = synergies.shape[1]
    n_dof = synergies.shape[2]

    # Compute the scale and shift matrix (correspond to H) and shifted and scaled synergies (correspond to W H)
    H = np.zeros((n_synergies, synergy_length, n_data, data_length))
    WH = np.zeros((n_dof, n_data, data_length))
    for n in range(n_data):
        for k in range(n_synergies):
            ts = delays[n, k]
            for i in range(synergy_length):
                H[k, i, n, ts+i] = amplitude[n, k]
                WH[:, n, ts+i] += synergies[k, i, :] * amplitude[n, k]

    # Reshape matrices
    data = np.transpose(data, (2, 0, 1))  # shape: (#DoF, #data, length)
    data = data.reshape((n_dof, n_data*data_length))  # shape: (#DoF, #data * length)
    synergies = np.transpose(synergies, (2, 0, 1))  # shape: (#DoF, #synergies, synergy-length)
    synergies = synergies.reshape((n_dof, n_synergies*synergy_length))  # shape: (#DoF, #synergies * synergy-length)
    H = H.reshape((n_synergies * synergy_length, n_data * data_length))  # shape: (#synergies * synergy-length, #data * length)
    WH = WH.reshape((n_dof, n_data*data_length))  # shape: (#DoF, #data * length)

    # Update synergies
    N = np.dot(data, H.T)
    D = np.dot(WH, H.T)
    #synergies = synergies * N / (D + eps)
    synergies = synergies + 0.001 * (N - D)

    synergies = synergies.reshape((n_dof, n_synergies, synergy_length))  # shape: (#DoF, #synergies, synergy-length)
    synergies = np.transpose(synergies, (1, 2, 0))  # shape: (#synergies, synergy-length, #DoF)

    return synergies
