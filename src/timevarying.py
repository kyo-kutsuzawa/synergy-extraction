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
    grad = np.einsum("ntm,ntmk->ntk", data - data_est, shifted_synergies)
    grad = -2 * np.sum(grad, axis=1)

    # Update the amplitude
    amplitude = amplitude - mu * grad
    amplitude = np.clip(amplitude, 0.0, None)  # Limit to non-negative values

    return amplitude


def update_synergies(data, synergies, amplitude, delays, mu=0.001):
    """Find synergies.

    The algorithm is based on [d'Avella and Tresch, 2002].
    """
    # Compute the gradient
    grad = np.zeros_like(synergies)
    for k in range(synergies.shape[0]):
        for n in range(data.shape[0]):
            ts = delays[n, k]
            grad[k, :, :] += (data[n, ts:ts+synergies.shape[1], :] - synergies[k, :, :] * amplitude[n, k]) * amplitude[n, k]

    # Compute the gradient
    grad = grad * -2

    # Update the amplitude
    synergies = synergies - mu * grad
    synergies = np.clip(synergies, 0.0, None)  # Limit to non-negative values

    return synergies
