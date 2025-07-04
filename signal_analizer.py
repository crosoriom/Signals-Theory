"""
Advanced Signal Processing, Communications, and Compression Class

This script defines a generic SignalAnalyzer class for filtering, analog/digital modulation,
quantization, compression, and analysis of 1D signal data.
"""

import math
import heapq
from collections import Counter

import numpy as np
from scipy.signal import welch, correlate, butter, filtfilt, hilbert, gaussian
from sklearn.cluster import KMeans


class SignalAnalyzer:
    """
    A generic class to encapsulate signal processing steps for any 1D signal.

    This class provides a comprehensive toolkit for analyzing and manipulating
    1D signals. It supports a chained-method interface for a fluid workflow,
    covering operations from basic filtering to advanced digital modulation and
    source coding.

    Key Features:
    - Time-based signal segmentation.
    - IIR filtering (lowpass, highpass, bandpass, bandstop).
    - Analog modulation (AM, DSB-SC, SSB-SC, PM, FM) and demodulation.
    - Digital modulation (ASK, PSK, QAM, FSK, GMSK, etc.) using a generated bitstream.
    - Signal quantization (uniform, μ-law, K-Means clustering).
    - Source coding (DPCM, Huffman compression).
    - Signal analysis (PSD, ACF, amplitude distribution, error metrics).
    """

    def __init__(self, signal_data: np.ndarray, fs: float):
        """
        Initializes the analyzer with a signal array and its sampling rate.

        Args:
            signal_data (np.ndarray): The 1D NumPy array containing the signal data.
            fs (float): The sampling frequency of the signal in Hz.

        Raises:
            TypeError: If signal_data is not a 1D NumPy array.
            ValueError: If fs is not a positive number.
        """
        if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1:
            raise TypeError("signal_data must be a 1D NumPy array.")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("fs (sampling frequency) must be a positive number.")

        self.fs = fs
        self.full_signal = signal_data
        self.full_times = np.arange(len(self.full_signal)) / self.fs

        # Initialize to analyze the entire signal by default
        self.signal = self.full_signal
        self.times = self.full_times
        self._reset_downstream_analyses()

        print("SignalAnalyzer initialized successfully.")

    def _reset_downstream_analyses(self):
        """
        Resets all calculated results that depend on the current signal or filter state.
        This is called when the underlying signal segment or filter is changed.
        """
        self.filtered_signal = None
        self.quantized_signal = None
        self.quantized_indices = None
        self.kmeans_model = None
        self.modulated_signal = None
        self.demodulated_signal = None
        self.bitstream = None
        self.baseband_i = None
        self.baseband_q = None

        self.filter_params = {}
        self.quantization_params = {}
        self.modulation_params = {}
        self.pulse_shaping_params = {}
        self.compression_results = {}
        self.error_metrics = {}
        self.histogram_results = {}
        self.psd_results = {}
        self.acf_results = {}

    def get_active_signal(self) -> np.ndarray:
        """
        Returns the current signal being analyzed.
        This will be the filtered signal if a filter has been applied,
        otherwise it will be the selected raw signal segment.

        Returns:
            np.ndarray: The active 1D signal array.
        """
        return self.filtered_signal if self.filtered_signal is not None else self.signal

    def select_time_interval(self, start_time: float, end_time: float):
        """
        Focuses all subsequent analysis on a sub-segment of the signal.
        This action resets any previously filtered signals or calculated results.

        Args:
            start_time (float): The start time of the segment in seconds.
            end_time (float): The end time of the segment in seconds.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            ValueError: If the time interval is invalid or out of bounds.
        """
        start_sample = int(start_time * self.fs)
        end_sample = int(end_time * self.fs)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.full_signal), end_sample)

        if start_sample >= end_sample:
            raise ValueError("Invalid time interval. Start must be before end and within bounds.")

        self.signal = self.full_signal[start_sample:end_sample]
        self.times = self.full_times[start_sample:end_sample]

        # Reset results since the base signal has changed
        self._reset_downstream_analyses()

        print(f"Analysis focused on time interval: {self.times[0]:.2f}s to {self.times[-1]:.2f}s")
        return self

    # --- FILTERING METHODS ---

    def _apply_filter(self, b, a):
        """
        Internal helper to apply a filter and reset downstream analysis.

        Args:
            b (np.ndarray): The numerator coefficient vector of the filter.
            a (np.ndarray): The denominator coefficient vector of the filter.
        """
        # A new filter invalidates all subsequent analyses.
        self._reset_downstream_analyses()
        # Use filtfilt for zero-phase filtering, which avoids phase distortion.
        self.filtered_signal = filtfilt(b, a, self.signal)

    def apply_lowpass_filter(self, cutoff: float, order: int = 4):
        """
        Applies a Butterworth low-pass filter to the active signal.

        Args:
            cutoff (float): The cutoff frequency in Hz.
            order (int): The order of the filter.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, cutoff / nyquist, btype='low')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'lowpass', 'cutoff': cutoff, 'order': order}
        print(f"Applied a low-pass filter at {cutoff} Hz.")
        return self

    def apply_highpass_filter(self, cutoff: float, order: int = 4):
        """
        Applies a Butterworth high-pass filter to the active signal.

        Args:
            cutoff (float): The cutoff frequency in Hz.
            order (int): The order of the filter.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, cutoff / nyquist, btype='high')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'highpass', 'cutoff': cutoff, 'order': order}
        print(f"Applied a high-pass filter at {cutoff} Hz.")
        return self

    def apply_bandpass_filter(self, lowcut: float, highcut: float, order: int = 4):
        """
        Applies a Butterworth band-pass filter to the active signal.

        Args:
            lowcut (float): The lower cutoff frequency in Hz.
            highcut (float): The higher cutoff frequency in Hz.
            order (int): The order of the filter.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='bandpass')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'bandpass', 'lowcut': lowcut, 'highcut': highcut, 'order': order}
        print(f"Applied a band-pass filter from {lowcut} to {highcut} Hz.")
        return self

    def apply_bandstop_filter(self, lowcut: float, highcut: float, order: int = 4):
        """
        Applies a Butterworth band-stop (reject-band/notch) filter.

        Args:
            lowcut (float): The lower cutoff frequency in Hz.
            highcut (float): The higher cutoff frequency in Hz.
            order (int): The order of the filter.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='bandstop')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'bandstop', 'lowcut': lowcut, 'highcut': highcut, 'order': order}
        print(f"Applied a band-stop filter from {lowcut} to {highcut} Hz.")
        return self

    # --- ANALOG MODULATION / DEMODULATION METHODS ---

    def _generate_carrier(self, carrier_freq: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates cosine and sine carrier waves for modulation.

        Args:
            carrier_freq (float): The carrier frequency in Hz.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the cosine (I) and sine (Q) carrier signals.
        """
        return np.cos(2 * np.pi * carrier_freq * self.times), np.sin(2 * np.pi * carrier_freq * self.times)

    def modulate_am(self, carrier_freq: float, modulation_index: float = 0.5):
        """
        Modulates the active signal using Amplitude Modulation (AM).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            modulation_index (float): The modulation index (depth). Must be > 0.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        m_t = self.get_active_signal()
        # Normalize message signal to make modulation_index meaningful and prevent over-modulation.
        m_t_norm = m_t / np.max(np.abs(m_t))
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = (1 + modulation_index * m_t_norm) * carrier_cos
        self.modulation_params = {'type': 'AM', 'carrier_freq': carrier_freq, 'mod_index': modulation_index}
        print(f"Applied AM with fc={carrier_freq} Hz.")
        return self

    def modulate_dsb_sc(self, carrier_freq: float):
        """
        Modulates the active signal using Double-Sideband Suppressed-Carrier (DSB-SC).

        Args:
            carrier_freq (float): The carrier frequency in Hz.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        m_t = self.get_active_signal()
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = m_t * carrier_cos
        self.modulation_params = {'type': 'DSB-SC', 'carrier_freq': carrier_freq}
        print(f"Applied DSB-SC with fc={carrier_freq} Hz.")
        return self

    def modulate_ssb_sc(self, carrier_freq: float, side: str = 'upper'):
        """
        Modulates using Single-Sideband Suppressed-Carrier (SSB-SC) via Hilbert transform.

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            side (str): The desired sideband, either 'upper' or 'lower'.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            ValueError: If `side` is not 'upper' or 'lower'.
        """
        if side not in ['upper', 'lower']:
            raise ValueError("Side must be 'upper' or 'lower'.")
        m_t = self.get_active_signal()
        # The Hilbert transform generates the quadrature component of the message signal.
        m_h = np.imag(hilbert(m_t))
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        if side == 'upper':
            self.modulated_signal = m_t * carrier_cos - m_h * carrier_sin
        else:  # lower
            self.modulated_signal = m_t * carrier_cos + m_h * carrier_sin
        self.modulation_params = {'type': 'SSB-SC', 'carrier_freq': carrier_freq, 'side': side}
        print(f"Applied {side}-sideband SSB-SC with fc={carrier_freq} Hz.")
        return self

    def modulate_pm(self, carrier_freq: float, deviation: float = np.pi/2):
        """
        Modulates the active signal using Phase Modulation (PM).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            deviation (float): The maximum phase shift (sensitivity) in radians.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        m_t = self.get_active_signal()
        m_t_norm = m_t / np.max(np.abs(m_t))
        carrier_phase = 2 * np.pi * carrier_freq * self.times
        self.modulated_signal = np.cos(carrier_phase + deviation * m_t_norm)
        self.modulation_params = {'type': 'PM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied PM with fc={carrier_freq} Hz.")
        return self

    def modulate_fm(self, carrier_freq: float, deviation: float = 50.0):
        """
        Modulates the active signal using Frequency Modulation (FM).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            deviation (float): The maximum frequency deviation (sensitivity) in Hz.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        m_t = self.get_active_signal()
        m_t_norm = m_t / np.max(np.abs(m_t))
        # Frequency is the derivative of phase, so phase is the integral of frequency.
        # Here we integrate the message signal to get the phase deviation.
        phase_dev = 2 * np.pi * deviation * np.cumsum(m_t_norm) / self.fs
        carrier_phase = 2 * np.pi * carrier_freq * self.times
        self.modulated_signal = np.cos(carrier_phase + phase_dev)
        self.modulation_params = {'type': 'FM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied FM with fc={carrier_freq} Hz.")
        return self

    def demodulate_coherent(self, lpf_cutoff: float = None):
        """
        Demodulates AM, DSB-SC, or SSB-SC using a coherent receiver.

        For SSB, this assumes a perfectly phase-locked local oscillator.

        Args:
            lpf_cutoff (float, optional): The cutoff frequency for the low-pass filter.
                                          If None, a default of 0.5 * carrier_freq is used.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If no modulated signal exists to demodulate.
        """
        if self.modulated_signal is None:
            raise RuntimeError("No modulated signal to demodulate.")
        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']
        carrier_cos, _ = self._generate_carrier(fc)

        # Mix the received signal with the local carrier.
        v_t = s_t * carrier_cos

        # Design a low-pass filter to remove the 2*fc component.
        if lpf_cutoff is None:
            lpf_cutoff = fc * 0.5  # A reasonable default to capture the baseband.
        nyquist = 0.5 * self.fs
        b, a = butter(5, lpf_cutoff / nyquist, btype='low')
        self.demodulated_signal = filtfilt(b, a, v_t)
        print(f"Applied coherent demodulation with LPF at {lpf_cutoff} Hz.")
        return self

    def demodulate_angle(self):
        """
        Demodulates PM or FM signals using a phase-based (Hilbert) method.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If the signal to demodulate is not a PM or FM signal.
        """
        if self.modulated_signal is None:
            raise RuntimeError("No modulated signal to demodulate.")
        mod_type = self.modulation_params.get('type')
        if mod_type not in ['PM', 'FM']:
            raise RuntimeError("Angle demodulation is only suitable for PM or FM signals.")

        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']

        # Use the Hilbert transform to create an analytic signal, from which
        # we can extract the instantaneous phase.
        analytic_signal = hilbert(s_t)
        # np.unwrap handles phase jumps greater than 2*pi.
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        if mod_type == 'PM':
            # For PM, the message is proportional to the phase deviation.
            carrier_phase = 2 * np.pi * fc * self.times
            self.demodulated_signal = (instantaneous_phase - carrier_phase)
        else:  # FM
            # For FM, the message is proportional to the frequency deviation,
            # which is the derivative of the phase.
            instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi / self.fs)
            # Match original signal length and subtract carrier frequency.
            self.demodulated_signal = np.append(instantaneous_freq, instantaneous_freq[-1]) - fc

        print(f"Applied angle demodulation for {mod_type}.")
        return self

    # --- QUANTIZATION METHODS ---

    def quantize_uniform(self, num_levels: int):
        """
        Quantizes the active signal using uniform step sizes.

        Args:
            num_levels (int): The number of discrete levels for quantization.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        signal_to_quantize = self.get_active_signal()
        min_val, max_val = np.min(signal_to_quantize), np.max(signal_to_quantize)

        if max_val == min_val: # Handle case of a flat signal
            self.quantized_signal = np.full_like(signal_to_quantize, min_val)
            self.quantized_indices = np.zeros_like(signal_to_quantize, dtype=int)
        else:
            step_size = (max_val - min_val) / num_levels
            # Determine the index for each sample.
            indices = np.floor((signal_to_quantize - min_val) / step_size)
            self.quantized_indices = np.clip(indices, 0, num_levels - 1).astype(int)
            # Reconstruct the signal from the center of each quantization bin.
            reconstructed_levels = min_val + self.quantized_indices * step_size + (step_size / 2)
            self.quantized_signal = reconstructed_levels

        self.quantization_params = {
            'type': 'uniform', 'num_levels': num_levels,
            'min_val': min_val, 'max_val': max_val
        }
        print(f"Applied uniform quantization with {num_levels} levels.")
        return self

    def quantize_non_uniform_mu_law(self, num_levels: int, mu: int = 255):
        """
        Quantizes the active signal using μ-law companding followed by uniform quantization.
        This gives more resolution to low-amplitude signals.

        Args:
            num_levels (int): The number of discrete levels for quantization.
            mu (int): The μ-law compression parameter. Typically 255 for 8-bit audio.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        signal_to_quantize = self.get_active_signal()
        max_abs_val = np.max(np.abs(signal_to_quantize))
        if max_abs_val == 0:
            self.quantized_signal = np.zeros_like(signal_to_quantize)
            self.quantized_indices = np.zeros_like(signal_to_quantize, dtype=int)
            return self

        # Compress the signal using the mu-law formula
        norm_signal = signal_to_quantize / max_abs_val
        comp_signal = np.sign(norm_signal) * np.log(1 + mu * np.abs(norm_signal)) / np.log(1 + mu)

        # Uniformly quantize the compressed signal (which is now in range [-1, 1])
        step_comp = 2.0 / num_levels
        indices = np.floor((comp_signal - (-1)) / step_comp)
        self.quantized_indices = np.clip(indices, 0, num_levels - 1).astype(int)

        # Reconstruct and expand to get the final quantized signal
        recon_comp = -1 + self.quantized_indices * step_comp + (step_comp / 2)
        exp_signal = np.sign(recon_comp) * ((1 + mu)**np.abs(recon_comp) - 1) / mu
        self.quantized_signal = exp_signal * max_abs_val

        self.quantization_params = {
            'type': 'mu-law', 'num_levels': num_levels, 'mu': mu, 'max_abs_val': max_abs_val
        }
        print(f"Applied μ-law quantization with {num_levels} levels.")
        return self

    def quantize_non_uniform_kmeans(self, num_levels: int):
        """
        Quantizes the active signal using K-Means clustering to find optimal level boundaries.
        This is a data-driven approach that adapts to the signal's probability distribution.

        Args:
            num_levels (int): The number of clusters (quantization levels).

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        signal_to_quantize = self.get_active_signal()
        # n_init='auto' is recommended to avoid FutureWarning in scikit-learn
        kmeans = KMeans(n_clusters=num_levels, n_init='auto', random_state=42)
        kmeans.fit(signal_to_quantize.reshape(-1, 1))

        self.kmeans_model = kmeans
        levels = kmeans.cluster_centers_.flatten()
        indices = kmeans.labels_

        # Sort the levels and create a mapping so that index 0 corresponds to the
        # lowest level, index 1 to the next lowest, and so on.
        sorted_level_indices = np.argsort(levels)
        sorted_levels = levels[sorted_level_indices]
        remap_array = np.zeros_like(sorted_level_indices)
        remap_array[sorted_level_indices] = np.arange(num_levels)

        self.quantized_indices = remap_array[indices]
        self.quantized_signal = sorted_levels[self.quantized_indices]
        self.quantization_params = {'type': 'kmeans', 'num_levels': num_levels, 'levels': sorted_levels}
        print(f"Applied K-Means quantization with {num_levels} levels.")
        return self

    # --- DISTRIBUTION ANALYSIS METHOD ---

    def calculate_histogram(self, bins: int = 100):
        """
        Calculates the amplitude distribution histogram of the active signal.

        Args:
            bins (int): The number of bins to use for the histogram.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        signal_to_analyze = self.get_active_signal()
        counts, bin_edges = np.histogram(signal_to_analyze, bins=bins, density=True)

        self.histogram_results = {
            'counts': counts,
            'bin_edges': bin_edges,
            'bins': bins
        }
        print(f"Calculated histogram with {bins} bins.")
        return self

    # --- DIGITAL DATA GENERATION AND PULSE SHAPING ---

    def generate_bitstream_from_indices(self):
        """
        Converts quantized indices into a single binary string (bitstream).

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If quantization has not been applied first.
        """
        if self.quantized_indices is None:
            raise RuntimeError("Quantization must be applied first to generate indices.")
        num_levels = self.quantization_params.get('num_levels')
        if num_levels is None:
            raise RuntimeError("Quantization parameters not found.")

        bits_per_symbol = math.ceil(math.log2(num_levels))
        self.bitstream = "".join([np.binary_repr(idx, width=bits_per_symbol) for idx in self.quantized_indices])
        print(f"Generated bitstream with {len(self.bitstream)} bits ({bits_per_symbol} bits/symbol).")
        return self

    def _generate_baseband_from_bits(self, samples_per_symbol: int, bits_per_symbol: int = 1) -> np.ndarray:
        """
        Helper to create a baseband signal from the bitstream by mapping bits to symbols.

        Args:
            samples_per_symbol (int): The number of samples to represent one symbol.
            bits_per_symbol (int): The number of bits per symbol (e.g., 1 for BPSK, 2 for QPSK).

        Returns:
            np.ndarray: An array of integer symbols.
        """
        if self.bitstream is None:
            self.generate_bitstream_from_indices()

        # Pad bitstream to be a multiple of bits_per_symbol for correct grouping.
        rem = len(self.bitstream) % bits_per_symbol
        bits_padded = self.bitstream + '0' * ((bits_per_symbol - rem) % bits_per_symbol)

        # Convert bit string chunks to integer symbols.
        symbols = np.array([int(bits_padded[i:i+bits_per_symbol], 2)
                           for i in range(0, len(bits_padded), bits_per_symbol)])

        # Create a basic rectangular pulse train for the I-channel.
        # This will be overwritten by PSK/QAM mappers but is used by FSK/ASK.
        self.baseband_i = np.repeat(symbols, samples_per_symbol)
        self.baseband_q = None  # Q-channel not used by default.
        return symbols

    def apply_gaussian_pulse_shaping(self, samples_per_symbol: int, bt_product: float = 0.5):
        """
        Applies a Gaussian filter to the I and Q baseband signals to reduce spectral width.

        Args:
            samples_per_symbol (int): The number of samples per symbol period (T).
            bt_product (float): The bandwidth-time product of the Gaussian filter.
                                A lower value gives more smoothing and less bandwidth.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If a baseband signal has not been generated yet.
        """
        if self.baseband_i is None:
            raise RuntimeError("Baseband signal must be generated before pulse shaping.")

        # Create the Gaussian filter kernel.
        # Span is chosen to capture most of the filter's energy.
        t = np.arange(-samples_per_symbol * 2, samples_per_symbol * 2 + 1)
        # Standard deviation is derived from the BT product.
        std_dev = samples_per_symbol / (math.sqrt(2 * math.log(2)) * (1 / bt_product))
        gauss_filter = gaussian(len(t), std=std_dev)
        gauss_filter /= np.sum(gauss_filter)  # Normalize for unity gain.

        # Apply filter via convolution.
        self.baseband_i = np.convolve(self.baseband_i, gauss_filter, mode='same')
        if self.baseband_q is not None:
            self.baseband_q = np.convolve(self.baseband_q, gauss_filter, mode='same')

        self.pulse_shaping_params = {'type': 'gaussian', 'BT': bt_product}
        print(f"Applied Gaussian pulse shaping with BT={bt_product}.")
        return self

    # --- DIGITAL MODULATION METHODS ---

    def modulate_ask(self, carrier_freq: float, samples_per_symbol: int):
        """
        Modulates the bitstream using Amplitude Shift Keying (On-Off Keying for binary).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol/bit.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = self.baseband_i * carrier_cos[:len(self.baseband_i)]
        self.modulation_params = {'type': 'ASK', 'carrier_freq': carrier_freq, 'M': 2}
        print(f"Applied ASK (OOK) modulation.")
        return self

    def modulate_psk(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """
        Modulates the bitstream using M-ary Phase Shift Keying (BPSK, QPSK, 8-PSK, etc.).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol.
            M (int): The number of phases (e.g., 2 for BPSK, 4 for QPSK). Must be a power of 2.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)

        # Map integer symbols to phase shifts.
        phases = 2 * np.pi * symbols / M
        self.baseband_i = np.repeat(np.cos(phases), samples_per_symbol)
        self.baseband_q = np.repeat(np.sin(phases), samples_per_symbol)

        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]

        mod_type = {2: 'BPSK', 4: 'QPSK'}.get(M, f'{M}-PSK')
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_oqpsk(self, carrier_freq: float, samples_per_symbol: int):
        """
        Modulates the bitstream using Offset QPSK (OQPSK).

        This is a variant of QPSK where the Q-channel is delayed by half a symbol period
        to prevent 180-degree phase shifts, reducing amplitude fluctuations.

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        # Start with a standard QPSK signal.
        self.modulate_psk(carrier_freq, samples_per_symbol, M=4)
        
        # Offset the Q channel by half a symbol period.
        offset = samples_per_symbol // 2
        self.baseband_q = np.roll(self.baseband_q, offset)
        
        # Re-modulate with the offset Q channel.
        len_match = len(self.baseband_i)
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params['type'] = 'OQPSK'
        print(f"Applied OQPSK modulation.")
        return self

    def modulate_fsk(self, carrier_freq: float, samples_per_symbol: int, M: int, freq_sep: float):
        """
        Modulates the bitstream using M-ary Frequency Shift Keying.

        Args:
            carrier_freq (float): The center carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol.
            M (int): The number of frequencies. Must be a power of 2.
            freq_sep (float): The separation between adjacent frequencies in Hz.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)

        # Generate M distinct carrier frequencies centered around carrier_freq.
        freqs = carrier_freq + np.arange(-(M-1)/2, (M-1)/2 + 1) * freq_sep
        
        # Create a signal where the instantaneous frequency changes based on the symbol.
        inst_freq = np.repeat(freqs[symbols], samples_per_symbol)
        # Integrate frequency over time to get phase.
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.fs
        self.modulated_signal = np.cos(phase)

        mod_type = 'BFSK' if M == 2 else f'{M}-FSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M, 'freq_sep': freq_sep}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_cpm(self, carrier_freq: float, samples_per_symbol: int, h: float, L: int = 1):
        """
        Modulates using Continuous Phase Modulation (engine for CPFSK, MSK).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol (T).
            h (float): The modulation index. h=0.5 for Minimum Shift Keying (MSK).
            L (int): The length of the frequency pulse in symbol periods. L=1 for rectangular (1REC).

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        # Map binary symbols {0, 1} to bipolar { -1, 1 }.
        alpha = 2 * symbols - 1

        # Create a rectangular frequency pulse g(t) of length L*T.
        g_t = np.ones(L * samples_per_symbol) / (2 * L * samples_per_symbol)
        # Convolve with symbols to get a continuous phase-change signal.
        phase_pulses = np.convolve(np.repeat(alpha, samples_per_symbol), g_t, mode='full')

        # Integrate the phase-change signal to get the final phase.
        phase = np.pi * h * np.cumsum(phase_pulses) / self.fs
        carrier_phase = 2 * np.pi * carrier_freq * np.arange(len(phase)) / self.fs
        self.modulated_signal = np.cos(carrier_phase + phase)

        mod_type = 'MSK' if h == 0.5 and L == 1 else 'CPFSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'h': h, 'L': L}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_gmsk(self, carrier_freq: float, samples_per_symbol: int, bt_product: float = 0.3):
        """
        Modulates the bitstream using Gaussian Minimum Shift Keying (GMSK).

        GMSK is a form of CPM where the frequency pulse is shaped with a Gaussian filter
        before integration, resulting in a very smooth phase transition and excellent
        spectral efficiency.

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol.
            bt_product (float): The bandwidth-time product of the Gaussian filter.
                                A common value for GSM is 0.3.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        alpha = 2 * symbols - 1 # Bipolar symbols {-1, 1}

        # 1. Create a rectangular pulse train from symbols.
        rect_pulses = np.repeat(alpha, samples_per_symbol)

        # 2. Apply Gaussian shaping to the rectangular pulses.
        # This instance of the class is used only for the shaping logic.
        shaper = SignalAnalyzer(rect_pulses, self.fs)
        shaper.baseband_i = rect_pulses
        shaper.apply_gaussian_pulse_shaping(samples_per_symbol, bt_product)
        freq_pulses = shaper.baseband_i

        # 3. Integrate the shaped frequency pulse to get the phase (h=0.5 for MSK).
        phase = np.pi * 0.5 * np.cumsum(freq_pulses) / self.fs
        t = np.arange(len(phase)) / self.fs
        carrier_phase = 2 * np.pi * carrier_freq * t
        self.modulated_signal = np.cos(carrier_phase + phase)
        
        self.pulse_shaping_params = shaper.pulse_shaping_params
        self.modulation_params = {'type': 'GMSK', 'carrier_freq': carrier_freq, 'BT': bt_product}
        print(f"Applied GMSK modulation.")
        return self

    def modulate_qam(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """
        Modulates the bitstream using M-ary Quadrature Amplitude Modulation (QAM).

        Args:
            carrier_freq (float): The carrier frequency in Hz.
            samples_per_symbol (int): Number of samples per symbol.
            M (int): The number of constellation points (e.g., 4, 16, 64).
                     Must be a perfect square for this implementation.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            ValueError: If M is not a perfect square, as required for square QAM.
        """
        bits_per_symbol = int(np.log2(M))
        if bits_per_symbol % 2 != 0:
            raise ValueError("For square QAM, M must be a power of 4 (4, 16, 64...).")

        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)

        # Map integer symbols to complex constellation points (I and Q amplitudes).
        # This example uses a standard Gray-coded 16-QAM map.
        if M == 16:
            qam_map = { # Gray-coded mapping
                0: (-3+3j),  1: (-3+1j),  2: (-3-3j),  3: (-3-1j),
                4: (-1+3j),  5: (-1+1j),  6: (-1-3j),  7: (-1-1j),
                8: (3+3j),   9: (3+1j),  10: (3-3j),  11: (3-1j),
                12: (1+3j),  13: (1+1j), 14: (1-3j),  15: (1-1j)
            }
            complex_symbols = np.array([qam_map.get(s, 0+0j) for s in symbols])
        else: # Generic square QAM mapping (not Gray-coded)
            sqrt_M = int(np.sqrt(M))
            i_levels = (2 * (symbols % sqrt_M) - (sqrt_M - 1))
            q_levels = (2 * (symbols // sqrt_M) - (sqrt_M - 1))
            complex_symbols = i_levels + 1j * q_levels

        self.baseband_i = np.repeat(np.real(complex_symbols), samples_per_symbol)
        self.baseband_q = np.repeat(np.imag(complex_symbols), samples_per_symbol)

        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params = {'type': f'{M}-QAM', 'carrier_freq': carrier_freq, 'M': M}
        print(f"Applied {M}-QAM modulation.")
        return self

    # --- DIGITAL DEMODULATION ---
    def demodulate_coherent_digital(self, lpf_cutoff: float = None):
        """
        Coherent I/Q demodulator for ASK, PSK, and QAM signals.

        This method recovers the analog baseband I and Q signals. Further processing
        (sampling and slicing) would be needed to recover the digital bits.

        Args:
            lpf_cutoff (float, optional): The cutoff frequency for the low-pass filter.
                                          If None, a default of 0.9 * carrier_freq is used.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']

        # I/Q mixing branches
        carrier_cos, carrier_sin = self._generate_carrier(fc)
        len_match = len(s_t)
        # Multiply by 2 to compensate for the 1/2 factor from trigonometric identities.
        i_branch = s_t * carrier_cos[:len_match] * 2
        q_branch = s_t * (-carrier_sin[:len_match]) * 2

        # Low-pass filter to remove 2*fc components
        if lpf_cutoff is None:
            lpf_cutoff = fc * 0.9
        nyquist = 0.5 * self.fs
        b, a = butter(5, lpf_cutoff / nyquist, btype='low')
        i_demod = filtfilt(b, a, i_branch)
        q_demod = filtfilt(b, a, q_branch)

        # Store the recovered analog baseband signals.
        self.demodulated_signal = {'I': i_demod, 'Q': q_demod}
        print(f"Applied coherent digital demodulation.")
        return self

    # --- ERROR AND COMPRESSION METHODS ---

    def calculate_error_metrics(self):
        """
        Calculates error between the original active signal and its quantized version.

        Computes the following metrics:
        - Mean Squared Error (MSE): Average squared difference in amplitude.
        - Symbol Error Count: Number of samples assigned to a different quantization bin
          than their ideal bin.
        - Symbol Error Rate (SER): The fraction of symbols that were in error.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If quantization has not been applied first.
            NotImplementedError: If error calculation is not supported for the
                                 current quantization type.
        """
        if self.quantized_signal is None:
            raise RuntimeError("A quantization method must be applied before calculating error.")

        original_signal = self.get_active_signal()
        quant_params = self.quantization_params

        # 1. Calculate Mean Squared Error (MSE)
        mse = np.mean((original_signal - self.quantized_signal)**2)

        # 2. Calculate ideal indices for the original signal to find symbol errors.
        num_levels = quant_params['num_levels']
        if quant_params['type'] == 'uniform':
            min_val, max_val = quant_params['min_val'], quant_params['max_val']
            step_size = (max_val - min_val) / num_levels
            ideal_indices = np.floor((original_signal - min_val) / step_size)
            ideal_indices = np.clip(ideal_indices, 0, num_levels - 1).astype(int)

        elif quant_params['type'] == 'mu-law':
            max_abs_val, mu = quant_params['max_abs_val'], quant_params['mu']
            norm_signal = original_signal / max_abs_val
            comp_signal = np.sign(norm_signal) * np.log(1 + mu * np.abs(norm_signal)) / np.log(1 + mu)
            step_comp = 2.0 / num_levels
            ideal_indices = np.floor((comp_signal - (-1)) / step_comp)
            ideal_indices = np.clip(ideal_indices, 0, num_levels - 1).astype(int)

        elif quant_params['type'] == 'kmeans':
            if self.kmeans_model is None: raise RuntimeError("K-Means model not found.")
            ideal_indices_raw = self.kmeans_model.predict(original_signal.reshape(-1, 1))
            # Remap to sorted indices to match the quantization step's output.
            levels = self.kmeans_model.cluster_centers_.flatten()
            sorted_level_indices = np.argsort(levels)
            remap_array = np.zeros_like(sorted_level_indices)
            remap_array[sorted_level_indices] = np.arange(num_levels)
            ideal_indices = remap_array[ideal_indices_raw]

        else:
            raise NotImplementedError(f"Error calculation not implemented for '{quant_params['type']}' quantization.")

        symbol_error_count = np.sum(ideal_indices != self.quantized_indices)
        symbol_error_rate = symbol_error_count / len(original_signal)

        self.error_metrics = {
            'mse': mse,
            'symbol_error_count': symbol_error_count,
            'symbol_error_rate': symbol_error_rate
        }

        print("\n--- Quantization Error Analysis ---")
        print(f"Mean Squared Error (MSE):         {mse:.2e}")
        print(f"Symbol Error Count:               {symbol_error_count} / {len(original_signal)}")
        print(f"Symbol Error Rate (SER):          {symbol_error_rate:.4%}")
        return self

    def compress_huffman(self):
        """
        Compresses the quantized indices using Huffman coding and calculates the ratio.

        This is a lossless compression technique that assigns shorter codes to more
        frequent symbols.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.

        Raises:
            RuntimeError: If quantization has not been applied first.
        """
        if self.quantized_indices is None:
            raise RuntimeError("A quantization method must be applied before compression.")

        # 1. Calculate frequency of each quantization index.
        frequency = Counter(self.quantized_indices)
        if not frequency:
            print("Signal is empty, nothing to compress.")
            return self

        # 2. Build the Huffman tree using a priority queue (min-heap).
        # Heap stores [weight, [symbol, code_string], [symbol, code_string], ...]
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]  # Prepend '0' to codes in the "left" branch
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]  # Prepend '1' to codes in the "right" branch
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        # 3. Create the Huffman codebook (dictionary of symbol -> code).
        huffman_codes = dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))

        # 4. Encode the signal indices into a single bitstring.
        encoded_signal = "".join([huffman_codes[s] for s in self.quantized_indices])

        # 5. Calculate compression statistics.
        num_levels = self.quantization_params['num_levels']
        bits_per_symbol_fixed = math.ceil(math.log2(num_levels)) if num_levels > 1 else 1
        original_size_bits = len(self.quantized_indices) * bits_per_symbol_fixed
        compressed_size_bits = len(encoded_signal)
        compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else float('inf')

        self.compression_results = {
            'type': 'huffman',
            'codes': huffman_codes,
            'encoded_string_preview': encoded_signal[:100] + "...",
            'original_bits': original_size_bits,
            'compressed_bits': compressed_size_bits,
            'compression_ratio': compression_ratio
        }

        print("\n--- Huffman Compression Results ---")
        print(f"Original size (fixed-length): {original_size_bits} bits")
        print(f"Compressed size (Huffman):    {compressed_size_bits} bits")
        print(f"Compression Ratio:              {compression_ratio:.2f} : 1")
        return self

    def apply_dpcm(self, num_levels: int):
        """
        Applies Differential Pulse-Code Modulation (DPCM), a predictive source coder.

        Instead of quantizing the signal directly, DPCM quantizes the difference
        (error) between the current sample and a prediction of it. This is effective
        for signals where adjacent samples are correlated.

        Args:
            num_levels (int): The number of levels to use for quantizing the error signal.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        signal_to_code = self.get_active_signal()
        
        # Simple predictor: next sample will be like the last original sample.
        predictions = np.roll(signal_to_code, 1)
        predictions[0] = 0  # Initial prediction is zero.
        errors = signal_to_code - predictions

        # Quantize the error signal, not the original signal.
        error_quantizer = SignalAnalyzer(errors, self.fs)
        error_quantizer.quantize_uniform(num_levels)

        # The output indices are the indices of the quantized error.
        self.quantized_indices = error_quantizer.quantized_indices
        self.quantization_params = {'type': 'DPCM', 'num_levels': num_levels}

        # Reconstruct the signal to store as 'quantized_signal' for analysis.
        # The decoder must use the same predictor logic.
        reconstructed_errors = error_quantizer.quantized_signal
        reconstructed_signal = np.zeros_like(signal_to_code)
        # The reconstruction loop is necessary because each new sample depends on
        # the previously reconstructed one, not just the original.
        for i in range(1, len(signal_to_code)):
            # Prediction for next sample is the current reconstructed sample.
            prediction = reconstructed_signal[i-1]
            reconstructed_signal[i] = prediction + reconstructed_errors[i]
        self.quantized_signal = reconstructed_signal

        print(f"Applied DPCM with {num_levels} error quantization levels.")
        return self

    # --- SPECTRAL AND CORRELATION ANALYSIS ---

    def calculate_psd(self, nperseg: int = 2048):
        """
        Calculates Power Spectral Density (PSD) using Welch's method.

        The PSD is calculated for the original signal segment and the filtered
        signal, if available.

        Args:
            nperseg (int): Length of each segment for Welch's method.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        def _get_psd(signal_data):
            freqs, Pxx = welch(signal_data, self.fs, nperseg=nperseg)
            return freqs, Pxx

        self.psd_results['original'] = _get_psd(self.signal)
        if self.filtered_signal is not None:
            self.psd_results['filtered'] = _get_psd(self.filtered_signal)
        print("Calculated Power Spectral Density.")
        return self

    def calculate_acf(self):
        """
        Calculates the Autocorrelation Function (ACF).

        The ACF is calculated for the original signal segment and the filtered
        signal, if available. It is normalized to have a maximum value of 1.

        Returns:
            SignalAnalyzer: The instance itself for method chaining.
        """
        def _get_acf(signal_data):
            demeaned = signal_data - np.mean(signal_data)
            # 'full' mode returns correlations for all possible overlaps.
            acf = correlate(demeaned, demeaned, mode='full')
            # Normalize so the peak at zero lag is 1.
            acf /= acf[len(demeaned) - 1]
            lags = np.arange(-(len(signal_data) - 1), len(signal_data)) / self.fs
            return lags, acf

        self.acf_results['original'] = _get_acf(self.signal)
        if self.filtered_signal is not None:
            self.acf_results['filtered'] = _get_acf(self.filtered_signal)
        print("Calculated Autocorrelation Function.")
        return self
