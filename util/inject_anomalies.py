#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from tqdm import trange
from scipy.stats import bernoulli, norm, expon
from sklearn.neighbors import NearestNeighbors
from scipy.signal import fftconvolve, find_peaks
from scipy import interpolate


def _constant_timseries(T):
    return np.all(T == T[0])


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


class InjectAnomalies:

    def __init__(self,
                 random_state: int = 0,
                 verbose: bool = False,
                 max_window_size: int = 128,
                 min_window_size: int = 8,
                 noise_dist: str='norm'):
        """
        
        Parameters
        ----------
        random_state: int
            Random state
        
        verbose: bool
            Controls verbosity

        max_window_size: int
            Maximum window size of the anomaly

        min_window_size: int
            Minimum window size of the anomaly

        norm_dist: str
            Distribution of noise to inject. Can be one of `norm` or `exp`.
        """
        self.random_state = random_state
        np.random.seed(seed=self.random_state)
        self._VALID_ANOMALY_TYPES = [
            'spikes', 'contextual', 'flip', 'speedup', 'noise', 'cutoff',
            'scale', 'wander', 'average'
        ]
        self.verbose = verbose
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.noise_dist = noise_dist
        assert max_window_size > min_window_size, "Maximum window size must be greater than the minimum window size."

    def __str__(self):
        InjectAnomaliesObject = {
            'random_state': self.random_state,
            'anomaly_types': self._VALID_ANOMALY_TYPES,
            'verbosity': self.verbose,
            'max_window_size': self.max_window_size,
            'min_window_size': self.min_window_size,
        }
        return f'InjectAnomaliesObject: {InjectAnomaliesObject}'

    def compute_crosscorrelation(self, T):
        return np.corrcoef(T)

    def compute_window_size(self, T):
        T = (T - T.mean()).squeeze()
        autocorr = fftconvolve(T, T, mode='same')
        self.peaks, _ = find_peaks(autocorr, distance=self.min_window_size)
        if len(self.peaks) >1:
            window_size = min(
                max(int(np.diff(self.peaks).mean()), self.min_window_size),
                self.max_window_size)
        else:
            window_size = min(self.min_window_size, self.max_window_size)

        if self.verbose:
            print(f'Window size: {window_size}')
        return window_size

    def inject_spikes(self, n: int, p: float = 0.2, scale: float = 0.8):
        """Spike anomalies
        """
        mask = bernoulli.rvs(p=p, size=n)
        if self.noise_dist=='norm':
            spikes = norm.rvs(loc=0, scale=scale, size=n) 
        elif self.noise_dist=='exp':
            spikes = expon.rvs(loc=0, scale=scale, size=n)
        return mask * spikes
    
    def inject_noise(self, n: int, scale: float = 0.8):
        """Noise anomalies
        """
        if self.noise_dist=='norm':
            return np.random.normal(scale=scale, size=n)
        elif self.noise_dist=='exp':
            return np.random.exponential(scale=scale, size=n)

    def inject_contextual_anomaly(self,
                                  window: np.ndarray,
                                  scale: float = 0.8):
        """To inject an anomaly, compose a linear transformation (AX + b) with the time signal
        """
        if self.noise_dist=='norm':
            a = norm.rvs(loc=1, scale=scale)
            b = norm.rvs(loc=0, scale=scale)
        elif self.noise_dist=='exp':
            a = expon.rvs(loc=1, scale=scale)
            b = expon.rvs(loc=0, scale=scale)
        return a * window + b - window

    def compute_anomaly_properties(self, T):
        """Compute properties of the random anomaly
        """
        _, n_time = T.shape
        if len(np.unique(T))>1:
            self.estimated_window_size = self.compute_window_size(
                T[self.anomalous_feature, :])
            self.anomaly_length = np.random.randint(
                1, self.max_anomaly_length)
            self.anomaly_start = -1
            self.anomaly_end = n_time + 1
            n_peaks = len(self.peaks)
            if n_peaks > 1:
                first_peak_idx = self.peaks[np.random.randint(
                    np.ceil(0.1 * n_peaks),
                    n_peaks)]
            elif n_peaks == 1:
                first_peak_idx = self.peaks[0]
            else:
                first_peak_idx = np.random.randint(
                    self.estimated_window_size // 2, n_time -
                    self.estimated_window_size // 2)
            self.anomaly_start = max(
                first_peak_idx - self.estimated_window_size // 2, 0)
            self.anomaly_end = min(
                self.anomaly_start +
                self.anomaly_length * self.estimated_window_size, n_time)
        else:
            self.estimated_window_size = 1
            self.anomaly_length = np.random.randint(
                1, self.max_anomaly_length)
            self.anomaly_start = np.random.randint(0, n_time - self.anomaly_length)
            self.anomaly_end = self.anomaly_start + self.anomaly_length
        
        if self.verbose:
            print(
                f'Anomaly start: {self.anomaly_start} end: {self.anomaly_end}')

    def get_valid_anomaly_types(self):
        return self._VALID_ANOMALY_TYPES

    def get_default_anomaly_parameters(self):
        return self

    def set_random_anomaly_parameters(self):
        self.amplitude_scaling = np.random.choice([0.25, 0.3, 0.5, 2, 3, 4],
                                                  size=1).squeeze()
        self.speed = np.random.choice([0.25, 0.3, 0.5, 2, 3, 4],
                                      size=1).squeeze()
        self.constant_type = np.random.choice(
            ['quantile', 'noisy_0', 'noisy_1', '0', '1'], size=1).squeeze()
        self.baseline = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
                                         size=1).squeeze()

    def inject_anomalies(self,
                         T: np.ndarray,
                         anomaly_type='contextual',
                         random_parameters: bool = False,
                         max_anomaly_length: int = 4,
                         anomaly_size_type: float = 'mae',
                         feature_id: int = None,
                         correlation_scaling: int = 5,
                         anomaly_propensity: float = 0.5,
                         scale: float = 2,
                         speed: float = 2,
                         noise_std: float = 0.05,
                         amplitude_scaling: float = 2,
                         constant_type: str = 'noisy_0',
                         constant_quantile: float = 0.75,
                         baseline: float = 0.2,
                         ma_window: int = 2):
        """Function to inject different kinds of anomalies.
        
        Parameters
        ----------
        T: np.ndarray
            Timeseries
        
        anomaly_type: str
            Type of anomaly to inject. Can be one of spikes, contextual, flip, speedup, noise, 
            cutoff, scale and wander. 

        random_paramters: bool
            Whether to randomly initialize anomaly injection parameters
        
        feature_id: int
            The feature index to inject the anomaly at.

        max_anomaly_length: int
            Maximum length of anomaly in terms of windows. By default, 4.

        anomaly_size_type: float
            Metric to measure the size of an injected anomaly. Can be one of `mae`, 
            `nearest` and `mse`. By default, it is set to `mae`.

        correlation_scaling: int
            Parameter to scale the correlation between variables of a multi-variate timeseries. 
        
        anomaly_propensity: float
            Probability that we insert a spike anomaly at time t. 

        scale: float
            Standard deviation of the normal distribution which governs the 
            slope and intercept parameters of contextual and spike anomaly injection. 
        
        speed: float
            Parameter for `speed` anomaly injection. Slow down or fasten a window
            of timeseries. By default, increases frequency by a factor of 2. 
        
        noise_std: float
            Parameter for `noise` anomaly injection. Standard deviation of gaussian
            noise. By default, 0.05. 
        
        amplitude_scaling:float=2
            Parameter for `scale` anomaly injection. Scale the amplitude of a window 
            of timeseries. By default, scale the amplitude by a factor of 2.
        
        constant_type: str
            Parameter for `cutoff` anomaly injection. Replace a window of timeseries 
            to a constant value. Possible options include setting the window to the 
            a specific `quantile`, `0`, `1`, `noisy_0` and `noisy_1`.
        
        constant_quantile: float
            Parameter for `cutoff` anomaly injection. Quantile of the value to cutoff
            the values of the window. It is only considered when the constant type is
            `quantile`. By default, set to 0.75.

        baseline: float
            Parameter for `wander` anomaly injection. Induces a baseline wander. The 
            exent of the positive or negative elevation gain is specified by the 
            baseline parameter. By default, it is set to 0.2.
        """
        if random_parameters:
            self.set_random_anomaly_parameters()

        self.max_anomaly_length = max_anomaly_length
        self.speed = speed
        self.noise_std = noise_std
        self.constant_type = constant_type
        self.amplitude_scaling = amplitude_scaling
        self.constant_quantile = constant_quantile
        self.baseline = baseline
        self.ma_window = ma_window

        if anomaly_type not in self._VALID_ANOMALY_TYPES:
            raise ValueError(
                f'anomaly_type must be in {self._VALID_ANOMALY_TYPES} but {anomaly_type} was passed.'
            )

        if len(T.shape) == 1:
            n_features = 1
            T = T.reshape(1,-1)
        elif len(T.shape) == 2:
            n_features, _ = T.shape
        else:
            raise ValueError(
                f'Expected T to be a 1D or 2D array but got {len(T.shape)}D array.'
            )

        timeseries_with_anomalies = T.copy()

        if feature_id is None:
            if n_features == 1:
                self.anomalous_feature = 0
            else:
                features_with_signal = np.where(np.std(T, axis=1) > 0)[0]
                self.anomalous_feature = np.random.choice(features_with_signal)
        else:
            self.anomalous_feature = feature_id

        if self.verbose:
            print(f'Feature {self.anomalous_feature} has an anomaly!')


        if len(T) > 1:
            correlation_vec = self.compute_crosscorrelation(T)[self.anomalous_feature]
            correlation_vec = np.sign(correlation_vec) * np.power(np.abs(correlation_vec), 1 / correlation_scaling)
            correlation_vec[np.isnan(correlation_vec)] = 0
        else:
            correlation_vec = self.compute_crosscorrelation(T)
            correlation_vec = np.sign(correlation_vec) * np.power(np.abs(correlation_vec), 1 / correlation_scaling)
            if np.isnan(correlation_vec):
                correlation_vec = 1
                print('Correlation vector is nan. You may have constant time series.')
            correlation_vec = np.array([correlation_vec])
        if self.verbose:
            print(f'Correlation scaling vector: {correlation_vec}')

        self.compute_anomaly_properties(T)
        if self.verbose:
            print(
                f'Length of anomaly for flip, speedup, noise, cutoff, scale and wander anomalies: {self.anomaly_length}'
            )

        if anomaly_type == 'spikes':
            spikes = self.inject_spikes(n=T.shape[1],
                                        p=anomaly_propensity,
                                        scale=scale)
            timeseries_with_anomalies = timeseries_with_anomalies + np.tile(spikes, (len(T), 1)) * correlation_vec[:, None]
            timeseries_with_anomalies = np.clip(timeseries_with_anomalies, 0, None)
            if correlation_vec==[0]:
                spike_labels = np.array([0]*len(spikes))
            else:
                spike_labels = (np.abs(spikes) > 0).astype(int)


        elif anomaly_type == 'contextual':
            anomaly = self.inject_contextual_anomaly(
                T[:, self.anomaly_start:self.anomaly_end], scale=scale)
            timeseries_with_anomalies[:, self.anomaly_start:self.
                                      anomaly_end] = anomaly * correlation_vec[:,None] + T[:,self.anomaly_start:self.anomaly_end]

        elif anomaly_type == 'flip':
            for i in range(n_features):
                timeseries_with_anomalies[i, self.anomaly_start:self.anomaly_end] = \
                    timeseries_with_anomalies[i, self.anomaly_start:self.anomaly_end][::-1]

        elif anomaly_type == 'speedup':
            if self.verbose: print(f'Speed: {self.speed}')
            xnew = np.linspace(
                0, self.anomaly_end - self.anomaly_start - 1,
                int((self.anomaly_end - self.anomaly_start) / self.speed))
            ynew = np.zeros((n_features, len(xnew)))

            for i in range(n_features):
                interpf = interpolate.interp1d(
                    np.arange(self.anomaly_end - self.anomaly_start),
                    timeseries_with_anomalies[
                        i, self.anomaly_start:self.anomaly_end])
                ynew[i, :] = interpf(xnew)

            timeseries_with_anomalies = np.concatenate([
                timeseries_with_anomalies[:, :self.anomaly_start], ynew,
                timeseries_with_anomalies[:, self.anomaly_end:]
            ],
                                                       axis=1)
            anomaly_size = np.zeros(timeseries_with_anomalies.shape[1])
            anomaly_size[self.anomaly_start:self.anomaly_start +
                         ynew.shape[1]] = 1

        elif anomaly_type == 'noise':
            timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end]\
                     + self.inject_noise(n=self.anomaly_end-self.anomaly_start, scale=self.noise_std) 

        elif anomaly_type == 'cutoff':
            if self.verbose: print(f'Constant: {self.constant_type}')
            if self.constant_type == 'quantile':
                timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                    np.quantile(timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end], self.constant_quantile)
            elif self.constant_type == '0':
                timeseries_with_anomalies[:, self.anomaly_start:self.
                                          anomaly_end] = 0
            elif self.constant_type == '1':
                timeseries_with_anomalies[:, self.anomaly_start:self.
                                          anomaly_end] = 1
            elif self.constant_type == 'noisy_0':
                timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                    self.inject_noise(n=self.anomaly_end-self.anomaly_start, scale=0.01)
            elif self.constant_type == 'noisy_1':
                timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                    self.inject_noise(n=self.anomaly_end-self.anomaly_start, scale=0.01) + 1

        elif anomaly_type == 'scale':
            if self.verbose: print(f'Scale: {self.amplitude_scaling}')
            timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                self.amplitude_scaling*timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end]

        elif anomaly_type == 'wander':
            if self.verbose: print(f'Baseline: {self.baseline}')
            timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end] = \
                np.linspace(0, self.baseline, self.anomaly_end-self.anomaly_start) +\
                     timeseries_with_anomalies[:, self.anomaly_start:self.anomaly_end]
            timeseries_with_anomalies[:, self.anomaly_end:] = \
                timeseries_with_anomalies[:, self.anomaly_end:] + self.baseline

        elif anomaly_type == 'average':
            if self.verbose: print(f'Moving average window: {self.ma_window}')
            for i in range(n_features):
                timeseries_with_anomalies[i, self.anomaly_start:self.anomaly_end] = \
                    moving_average(timeseries_with_anomalies[i, self.anomaly_start:self.anomaly_end], self.ma_window)


        if anomaly_size_type == 'mae' and anomaly_type != 'speedup':
            anomaly_size = np.mean(np.abs(T - timeseries_with_anomalies),
                                   axis=0)
        elif anomaly_size_type == 'mse' and anomaly_type != 'speedup':
            anomaly_size = np.mean((T - timeseries_with_anomalies)**2, axis=0)
        elif anomaly_size_type == 'nearest' and anomaly_type != 'speedup':
            nearest = NearestNeighbors(n_neighbors=2,
                                       algorithm='ball_tree',
                                       metric='cityblock')
            nearest.fit(X=T.T)
            distances, _ = nearest.kneighbors(timeseries_with_anomalies.T)
            anomaly_size = distances.mean(axis=1) / len(T)

        anomaly_start = max(0, self.anomaly_start - self.min_window_size)
        if anomaly_type == 'speedup':
            anomaly_end = min(
                timeseries_with_anomalies.shape[1],
                self.anomaly_start + ynew.shape[1] + self.min_window_size)
        else:
            anomaly_end = min(timeseries_with_anomalies.shape[1],
                              self.anomaly_end + self.min_window_size)

        anomaly_labels = np.zeros(timeseries_with_anomalies.shape[1]).astype(int)
        anomaly_labels[anomaly_start:anomaly_end] = 1

        if anomaly_type == 'spikes':
            anomaly_start = np.where(spike_labels==1)[0][0]
            anomaly_end = np.where(spike_labels==1)[0][-1]
            anomaly_labels = np.zeros(timeseries_with_anomalies.shape[1]).astype(int)
            anomaly_labels[anomaly_start:anomaly_end] = 1

        return timeseries_with_anomalies, anomaly_size, anomaly_labels
