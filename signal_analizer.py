"""
Advanced Signal Processing, Communications, and Compression Class

This script defines a generic SignalAnalyzer class for filtering, analog/digital modulation,
quantization, compression, and analysis of 1D signal data.
"""

import numpy as np
import math
import heapq
from collections import Counter
from scipy.signal import welch, correlate, butter, filtfilt, hilbert
from sklearn.cluster import KMeans

class SignalAnalyzer:
    """
    A generic class to encapsulate signal processing steps for any 1D signal.
    It supports various filter types and calculates spectral and correlation properties.
    """
    def __init__(self, signal_data: np.ndarray, fs: float):
        """
        Initializes the analyzer with a signal array and its sampling rate.

        Args:
            signal_data (np.ndarray): The 1D NumPy array containing the signal data.
            fs (float): The sampling frequency of the signal in Hz.
        """
        if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1:
            raise TypeError("signal_data must be a 1D NumPy array.")
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("fs (sampling frequency) must be a positive number.")
        
        self.fs = fs
        self.full_signal = signal_data
        self.full_times = np.arange(len(self.full_signal)) / self.fs
        # Initialize state attributes to analyze the entire signal by default
        self._reset_downstream_analyses()
            
        print("SignalAnalyzer initialized successfully.")

    def get_active_signal(self):
        """Returns the filtered signal if available, otherwise the original signal segment."""
        return self.filtered_signal if self.filtered_signal is not None else self.signal

    def select_time_interval(self, start_time: float, end_time: float):
        """
        Focuses analysis on a sub-segment of the signal.
        This resets any previously filtered signals or calculated results.

        Args:
            start_time (float): The start time of the segment in seconds.
            end_time (float): The end time of the segment in seconds.
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
        # Reset all subsequent analysis results
        self._reset_downstream_analyses()
        self.filtered_signal = filtfilt(b, a, self.signal)

    def _reset_downstream_analyses(self):
        """Resets all results that depend on the current signal state."""
        self.filtered_signal = self.quantized_signal = self.quantized_indices = self.kmeans_model = None
        self.modulated_signal = self.demodulated_signal = None
        self.bitstream = self.baseband_i = self.baseband_q = None
        self.quantization_params = self.modulation_params = self.pulse_shaping_params = {}
        self.compression_results = self.error_metrics = self.histogram_results = {}
        self.psd_results = {}
        self.acf_results = {}

    def apply_lowpass_filter(self, cutoff: float, order: int = 4):
        """
        Applies a Butterworth low-pass filter.

        Args:
            cutoff (float): The cutoff frequency in Hz.
            order (int): The order of the filter.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, cutoff / nyquist, btype='low')
        self._apply_filter(b, a)
        self.quantization_params = {'type': 'lowpass', 'cutoff': cutoff}
        print(f"Applied a low-pass filter at {cutoff} Hz.")
        return self

    def apply_highpass_filter(self, cutoff: float, order: int = 4):
        """
        Applies a Butterworth high-pass filter.

        Args:
            cutoff (float): The cutoff frequency in Hz.
            order (int): The order of the filter.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, cutoff / nyquist, btype='high')
        self._apply_filter(b, a)
        self.quantization_params = {'type': 'highpass', 'cutoff': cutoff}
        print(f"Applied a high-pass filter at {cutoff} Hz.")
        return self

    def apply_bandpass_filter(self, lowcut: float, highcut: float, order: int = 4):
        """
        Applies a Butterworth band-pass filter.

        Args:
            lowcut (float): The lower cutoff frequency in Hz.
            highcut (float): The higher cutoff frequency in Hz.
            order (int): The order of the filter.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='bandpass')
        self._apply_filter(b, a)
        self.quantization_params = {'type': 'bandpass', 'lowcut': lowcut, 'highcut': highcut}
        print(f"Applied a band-pass filter from {lowcut} to {highcut} Hz.")
        return self

    def apply_bandstop_filter(self, lowcut: float, highcut: float, order: int = 4):
        """
        Applies a Butterworth band-stop (reject-band/notch) filter.

        Args:
            lowcut (float): The lower cutoff frequency in Hz.
            highcut (float): The higher cutoff frequency in Hz.
            order (int): The order of the filter.
        """
        nyquist = 0.5 * self.fs
        b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='bandstop')
        self._apply_filter(b, a)
        self.quantization_params = {'type': 'bandstop', 'lowcut': lowcut, 'highcut': highcut}
        print(f"Applied a band-stop filter from {lowcut} to {highcut} Hz.")
        return self

    # --- MODULATION / DEMODULATION METHODS ---

    def _generate_carrier(self, carrier_freq):
        """Generates sine and cosine carrier waves."""
        return np.cos(2 * np.pi * carrier_freq * self.times), np.sin(2 * np.pi * carrier_freq * self.times)

    def modulate_am(self, carrier_freq: float, modulation_index: float = 0.5):
        """Modulates the active signal using Amplitude Modulation (AM)."""
        m_t = self.get_active_signal()
        # Normalize message signal to make modulation_index meaningful
        m_t_norm = m_t / np.max(np.abs(m_t))
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = (1 + modulation_index * m_t_norm) * carrier_cos
        self.modulation_params = {'type': 'AM', 'carrier_freq': carrier_freq, 'mod_index': modulation_index}
        print(f"Applied AM with fc={carrier_freq} Hz.")
        return self

    def modulate_dsb_sc(self, carrier_freq: float):
        """Modulates using Double-Sideband Suppressed-Carrier."""
        m_t = self.get_active_signal()
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = m_t * carrier_cos
        self.modulation_params = {'type': 'DSB-SC', 'carrier_freq': carrier_freq}
        print(f"Applied DSB-SC with fc={carrier_freq} Hz.")
        return self

    def modulate_ssb_sc(self, carrier_freq: float, side: str = 'upper'):
        """Modulates using Single-Sideband Suppressed-Carrier (via Hilbert transform)."""
        if side not in ['upper', 'lower']: raise ValueError("Side must be 'upper' or 'lower'.")
        m_t = self.get_active_signal()
        m_h = np.imag(hilbert(m_t)) # Hilbert transform of the message
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        if side == 'upper':
            self.modulated_signal = m_t * carrier_cos - m_h * carrier_sin
        else: # lower
            self.modulated_signal = m_t * carrier_cos + m_h * carrier_sin
        self.modulation_params = {'type': 'SSB-SC', 'carrier_freq': carrier_freq, 'side': side}
        print(f"Applied {side}-sideband SSB-SC with fc={carrier_freq} Hz.")
        return self

    def modulate_pm(self, carrier_freq: float, deviation: float = np.pi/2):
        """Modulates using Phase Modulation (PM). `deviation` is the max phase shift in radians."""
        m_t = self.get_active_signal()
        m_t_norm = m_t / np.max(np.abs(m_t))
        carrier_cos_phase = 2 * np.pi * carrier_freq * self.times
        self.modulated_signal = np.cos(carrier_cos_phase + deviation * m_t_norm)
        self.modulation_params = {'type': 'PM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied PM with fc={carrier_freq} Hz.")
        return self

    def modulate_fm(self, carrier_freq: float, deviation: float = 50.0):
        """Modulates using Frequency Modulation (FM). `deviation` is max freq shift in Hz."""
        m_t = self.get_active_signal()
        m_t_norm = m_t / np.max(np.abs(m_t))
        # Integrate message signal to get phase deviation
        phase_dev = 2 * np.pi * deviation * np.cumsum(m_t_norm) / self.fs
        carrier_cos_phase = 2 * np.pi * carrier_freq * self.times
        self.modulated_signal = np.cos(carrier_cos_phase + phase_dev)
        self.modulation_params = {'type': 'FM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied FM with fc={carrier_freq} Hz.")
        return self

    def demodulate_coherent(self, lpf_cutoff: float = None):
        """Demodulates AM, DSB-SC, or SSB-SC using a coherent receiver."""
        if self.modulated_signal is None: raise RuntimeError("No modulated signal to demodulate.")
        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']
        carrier_cos, _ = self._generate_carrier(fc)
        
        v_t = s_t * carrier_cos # Multiply by carrier
        
        # Design a low-pass filter to remove 2fc component
        if lpf_cutoff is None: lpf_cutoff = fc * 0.5 # A reasonable default
        nyquist = 0.5 * self.fs
        b, a = butter(5, lpf_cutoff / nyquist, btype='low')
        self.demodulated_signal = filtfilt(b, a, v_t)
        print(f"Applied coherent demodulation with LPF at {lpf_cutoff} Hz.")
        return self

    def demodulate_angle(self):
        """Demodulates PM or FM signals using a phase-based method."""
        if self.modulated_signal is None: raise RuntimeError("No modulated signal to demodulate.")
        if self.modulation_params['type'] not in ['PM', 'FM']: raise RuntimeError("Angle demodulation only for PM/FM.")
        
        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']
        
        # Use Hilbert transform to find the instantaneous phase
        analytic_signal = hilbert(s_t)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        if self.modulation_params['type'] == 'PM':
            # Demodulated signal is the phase deviation from the carrier
            carrier_phase = 2 * np.pi * fc * self.times
            self.demodulated_signal = (instantaneous_phase - carrier_phase)
        else: # FM
            # Demodulated signal is the derivative of the phase (instantaneous frequency)
            instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi / self.fs)
            # Add a sample to match length and subtract carrier freq
            self.demodulated_signal = np.append(instantaneous_freq, instantaneous_freq[-1]) - fc
            
        print(f"Applied angle demodulation for {self.modulation_params['type']}.")
        return self

    # --- QUANTIZATION METHODS ---

    def quantize_uniform(self, num_levels: int):
        """Quantizes the active signal using uniform step sizes."""
        signal_to_quantize = self.get_active_signal()
        min_val, max_val = np.min(signal_to_quantize), np.max(signal_to_quantize)

        if max_val == min_val:
            self.quantized_signal = np.full_like(signal_to_quantize, min_val)
            self.quantized_indices = np.zeros_like(signal_to_quantize, dtype=int)
        else:
            step_size = (max_val - min_val) / num_levels
            # Get indices and clip to be safe
            indices = np.floor((signal_to_quantize - min_val) / step_size)
            self.quantized_indices = np.clip(indices, 0, num_levels - 1).astype(int)
            # Reconstruct signal from indices
            reconstructed_levels = min_val + self.quantized_indices * step_size + (step_size / 2)
            self.quantized_signal = reconstructed_levels

        self.quantization_params = {'type': 'uniform', 'num_levels': num_levels}
        print(f"Applied uniform quantization with {num_levels} levels.")
        return self

    def quantize_non_uniform_mu_law(self, num_levels: int, mu: int = 255):
        """Quantizes the active signal using μ-law companding."""
        signal_to_quantize = self.get_active_signal()
        max_abs_val = np.max(np.abs(signal_to_quantize))
        if max_abs_val == 0:
            self.quantized_signal = np.zeros_like(signal_to_quantize)
            self.quantized_indices = np.zeros_like(signal_to_quantize, dtype=int)
            return self

        norm_signal = signal_to_quantize / max_abs_val
        comp_signal = np.sign(norm_signal) * np.log(1 + mu * np.abs(norm_signal)) / np.log(1 + mu)
        
        # Uniformly quantize the compressed signal (range is -1 to 1)
        step_comp = 2.0 / num_levels
        indices = np.floor((comp_signal - (-1)) / step_comp)
        self.quantized_indices = np.clip(indices, 0, num_levels - 1).astype(int)
        
        # Reconstruct and expand
        recon_comp = -1 + self.quantized_indices * step_comp + (step_comp / 2)
        exp_signal = np.sign(recon_comp) * ((1 + mu)**np.abs(recon_comp) - 1) / mu
        self.quantized_signal = exp_signal * max_abs_val

        self.quantization_params = {'type': 'mu-law', 'num_levels': num_levels, 'mu': mu}
        print(f"Applied μ-law quantization with {num_levels} levels.")
        return self

    def quantize_non_uniform_kmeans(self, num_levels: int):
        """
        Quantizes the active signal using K-Means clustering to find optimal levels.
        """
        signal_to_quantize = self.get_active_signal()
        kmeans = KMeans(n_clusters=num_levels, n_init='auto', random_state=42).fit(signal_to_quantize.reshape(-1, 1))
        
        # Store the fitted model
        self.kmeans_model = kmeans 
        
        levels = kmeans.cluster_centers_.flatten()
        indices = kmeans.labels_
        sorted_level_indices = np.argsort(levels)
        sorted_levels = levels[sorted_level_indices]
        remap_array = np.zeros_like(sorted_level_indices); remap_array[sorted_level_indices] = np.arange(num_levels)
        self.quantized_indices = remap_array[indices]
        self.quantized_signal = sorted_levels[self.quantized_indices]
        self.quantization_params = {'type': 'kmeans', 'num_levels': num_levels, 'levels': sorted_levels}
        print(f"Applied K-Means quantization with {num_levels} levels.")
        return self

    # --- DISTRIBUTION ANALYSIS METHOD ---
    
    def calculate_histogram(self, bins: int = 100):
        """
        Calculates the histogram data for the amplitude distribution of the active signal.

        Args:
            bins (int): The number of bins to use for the histogram.
        """
        signal_to_analyze = self.get_active_signal()
        
        # Use NumPy's histogram function to get the counts and bin edges
        counts, bin_edges = np.histogram(signal_to_analyze, bins=bins)
        
        # Store the results in the class instance
        self.histogram_results = {
            'counts': counts,
            'bin_edges': bin_edges,
            'bins': bins
        }

        print(f"Calculated histogram with {bins} bins.")
        return self

    # --- DIGITAL DATA GENERATION AND PULSE SHAPING ---

    def generate_bitstream_from_indices(self):
        """Converts quantized indices into a single bitstream."""
        if self.quantized_indices is None: raise RuntimeError("Quantization must be applied first.")
        num_levels = self.quantization_params.get('num_levels')
        if num_levels is None: raise RuntimeError("Quantization parameters not found.")
        
        bits_per_symbol = math.ceil(math.log2(num_levels))
        self.bitstream = "".join([np.binary_repr(idx, width=bits_per_symbol) for idx in self.quantized_indices])
        print(f"Generated bitstream with {len(self.bitstream)} bits ({bits_per_symbol} bits/symbol).")
        return self

    def _generate_baseband_from_bits(self, samples_per_symbol, bits_per_symbol=1):
        """Helper to create I/Q baseband signals from the bitstream."""
        if self.bitstream is None: self.generate_bitstream_from_indices()
        
        # Pad bitstream to be a multiple of bits_per_symbol
        rem = len(self.bitstream) % bits_per_symbol
        bits_padded = self.bitstream + '0' * ((bits_per_symbol - rem) % bits_per_symbol)
        
        # Convert bit string to integer symbols
        symbols = np.array([int(bits_padded[i:i+bits_per_symbol], 2) for i in range(0, len(bits_padded), bits_per_symbol)])
        
        # Default mapping (e.g., for FSK)
        self.baseband_i = np.repeat(symbols, samples_per_symbol)
        self.baseband_q = None # Not used for all modulations
        return symbols

    def apply_gaussian_pulse_shaping(self, samples_per_symbol: int, bt_product: float = 0.5):
        """Applies a Gaussian filter to the I and Q baseband signals."""
        if self.baseband_i is None: raise RuntimeError("Baseband signal not generated.")
        
        # Create Gaussian filter kernel
        t = np.arange(-samples_per_symbol * 2, samples_per_symbol * 2 + 1)
        std_dev = samples_per_symbol / (math.sqrt(2 * math.log(2)) * (1/bt_product))
        gauss_filter = gaussian(len(t), std=std_dev)
        gauss_filter /= np.sum(gauss_filter) # Normalize
        
        # Apply filter via convolution
        self.baseband_i = np.convolve(self.baseband_i, gauss_filter, mode='same')
        if self.baseband_q is not None:
            self.baseband_q = np.convolve(self.baseband_q, gauss_filter, mode='same')
            
        self.pulse_shaping_params = {'type': 'gaussian', 'BT': bt_product}
        print(f"Applied Gaussian pulse shaping with BT={bt_product}.")
        return self

    # --- DIGITAL MODULATION METHODS ---

    def modulate_ask(self, carrier_freq: float, samples_per_symbol: int):
        """Amplitude Shift Keying (On-Off Keying for binary)."""
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = self.baseband_i * carrier_cos[:len(self.baseband_i)]
        self.modulation_params = {'type': 'ASK', 'carrier_freq': carrier_freq}
        return self

    def modulate_psk(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """Phase Shift Keying (BPSK, QPSK, 8-PSK, etc.)."""
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        
        # Map symbols to phase shifts
        phases = 2 * np.pi * symbols / M
        self.baseband_i = np.repeat(np.cos(phases), samples_per_symbol)
        self.baseband_q = np.repeat(np.sin(phases), samples_per_symbol)
        
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        
        if M == 2: mod_type = 'BPSK'
        elif M == 4: mod_type = 'QPSK'
        else: mod_type = f'{M}-PSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M}
        return self

    def modulate_oqpsk(self, carrier_freq: float, samples_per_symbol: int):
        """Offset QPSK."""
        self.modulate_psk(carrier_freq, samples_per_symbol, M=4)
        # Offset the Q channel by half a symbol period
        offset = samples_per_symbol // 2
        self.baseband_q = np.roll(self.baseband_q, offset)
        len_match = len(self.baseband_i)
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params['type'] = 'OQPSK'
        return self

    def modulate_fsk(self, carrier_freq: float, samples_per_symbol: int, M: int, freq_sep: float):
        """M-ary Frequency Shift Keying."""
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        
        # Generate M carrier frequencies
        freqs = carrier_freq + np.arange(-(M-1)/2, (M-1)/2 + 1) * freq_sep
        t = np.arange(len(symbols) * samples_per_symbol) / self.fs
        
        # Create frequency-modulated signal
        inst_freq = np.repeat(freqs[symbols], samples_per_symbol)
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.fs
        self.modulated_signal = np.cos(phase)
        
        mod_type = 'BFSK' if M == 2 else f'{M}-FSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M}
        return self

    def modulate_cpm(self, carrier_freq: float, samples_per_symbol: int, h: float, L: int=1):
        """General Continuous Phase Modulation engine (for CPFSK, MSK)."""
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        # Map binary symbols {0, 1} to {-1, 1}
        alpha = 2 * symbols - 1
        
        # Create rectangular frequency pulse g(t) of length L*T
        g_t = np.ones(L * samples_per_symbol) / (2 * L * samples_per_symbol)
        # Convolve with symbols to get phase pulse train
        phase_pulses = np.convolve(np.repeat(alpha, samples_per_symbol), g_t, mode='full')
        
        # Integrate to get phase
        phase = np.pi * h * np.cumsum(phase_pulses) / self.fs
        carrier_phase = 2 * np.pi * carrier_freq * np.arange(len(phase)) / self.fs
        self.modulated_signal = np.cos(carrier_phase + phase)
        
        mod_type = 'MSK' if h == 0.5 and L == 1 else 'CPFSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'h': h}
        return self

    def modulate_gmsk(self, carrier_freq: float, samples_per_symbol: int, bt_product: float = 0.3):
        """Gaussian Minimum Shift Keying."""
        self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        self.apply_gaussian_pulse_shaping(samples_per_symbol, bt_product)
        # Now use the shaped baseband as a frequency pulse for CPM
        alpha = 2 * self.baseband_i - 1
        phase = np.pi * 0.5 * np.cumsum(alpha) / self.fs
        t = np.arange(len(phase)) / self.fs
        carrier_phase = 2 * np.pi * carrier_freq * t
        self.modulated_signal = np.cos(carrier_phase + phase)
        self.modulation_params = {'type': 'GMSK', 'carrier_freq': carrier_freq, 'BT': bt_product}
        return self

    def modulate_qam(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """M-ary Quadrature Amplitude Modulation."""
        bits_per_symbol = int(np.log2(M))
        if bits_per_symbol % 2 != 0: raise ValueError("For square QAM, M must be a square number (4, 16, 64...).")
        
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        
        # Simple Gray coding for 16-QAM as an example
        if M == 16:
            qam_map = {
                0: (-3+3j),  1: (-3+1j),  2: (-3-3j),  3: (-3-1j),
                4: (-1+3j),  5: (-1+1j),  6: (-1-3j),  7: (-1-1j),
                8: (3+3j),   9: (3+1j),  10: (3-3j),  11: (3-1j),
                12: (1+3j),  13: (1+1j), 14: (1-3j),  15: (1-1j)
            }
            complex_symbols = np.array([qam_map[s] for s in symbols])
        else: # Fallback for other M
            sqrt_M = int(np.sqrt(M))
            i = (2 * (symbols % sqrt_M) - (sqrt_M - 1))
            q = (2 * (symbols // sqrt_M) - (sqrt_M - 1))
            complex_symbols = i + 1j*q

        self.baseband_i = np.repeat(np.real(complex_symbols), samples_per_symbol)
        self.baseband_q = np.repeat(np.imag(complex_symbols), samples_per_symbol)
        
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params = {'type': f'{M}-QAM', 'carrier_freq': carrier_freq, 'M': M}
        return self

    # --- DIGITAL DEMODULATION ---
    def demodulate_coherent_digital(self, samples_per_symbol: int, lpf_cutoff: float = None):
        """Coherent demodulator for ASK, PSK, QAM."""
        s_t = self.modulated_signal
        fc = self.modulation_params['carrier_freq']
        
        # I/Q branches
        carrier_cos, carrier_sin = self._generate_carrier(fc)
        len_match = len(s_t)
        i_branch = s_t * carrier_cos[:len_match] * 2
        q_branch = s_t * (-carrier_sin[:len_match]) * 2

        # Low-pass filter
        if lpf_cutoff is None: lpf_cutoff = fc * 0.9
        nyquist = 0.5 * self.fs
        b, a = butter(5, lpf_cutoff / nyquist, btype='low')
        i_demod = filtfilt(b, a, i_branch)
        q_demod = filtfilt(b, a, q_branch)
        
        # Sampler and slicer would go here to recover bits
        # For simplicity, we'll just store the analog demodulated baseband
        self.demodulated_signal = {'I': i_demod, 'Q': q_demod}
        print(f"Applied coherent digital demodulation.")
        return self

    # --- ERROR AND COMPRESSION METHODS ---

    def calculate_error_metrics(self):
        """
        Calculates the error between the original and quantized signals.
        - Mean Squared Error (MSE): Traditional measure of signal fidelity.
        - Hamming Distance / Symbol Error Rate: Measures how many samples were
          assigned to a different quantization bin.
        """
        if self.quantized_signal is None:
            raise RuntimeError("A quantization method must be applied before calculating error.")

        original_signal = self.get_active_signal()
        quant_params = self.quantization_params
        
        # 1. Calculate Mean Squared Error (MSE)
        mse = np.mean((original_signal - self.quantized_signal)**2)
        
        # 2. Calculate Hamming Distance by finding the "ideal" indices for the original signal
        if quant_params['type'] == 'uniform':
            min_val, max_val = quant_params['min_val'], quant_params['max_val']
            step_size = (max_val - min_val) / quant_params['num_levels']
            ideal_indices = np.floor((original_signal - min_val) / step_size)
            ideal_indices = np.clip(ideal_indices, 0, quant_params['num_levels'] - 1).astype(int)
        
        elif quant_params['type'] == 'mu-law':
            max_abs_val, mu, num_levels = quant_params['max_abs_val'], quant_params['mu'], quant_params['num_levels']
            norm_signal = original_signal / max_abs_val
            comp_signal = np.sign(norm_signal) * np.log(1 + mu * np.abs(norm_signal)) / np.log(1 + mu)
            step_comp = 2.0 / num_levels
            ideal_indices = np.floor((comp_signal - (-1)) / step_comp)
            ideal_indices = np.clip(ideal_indices, 0, num_levels - 1).astype(int)

        elif quant_params['type'] == 'kmeans':
            if self.kmeans_model is None: raise RuntimeError("K-Means model not found.")
            ideal_indices_raw = self.kmeans_model.predict(original_signal.reshape(-1, 1))
            # Remap to sorted indices just like in the quantization step
            levels = self.kmeans_model.cluster_centers_.flatten()
            sorted_level_indices = np.argsort(levels)
            remap_array = np.zeros_like(sorted_level_indices); remap_array[sorted_level_indices] = np.arange(quant_params['num_levels'])
            ideal_indices = remap_array[ideal_indices_raw]

        else:
            raise NotImplementedError("Hamming distance not implemented for this quantization type.")

        hamming_distance = np.sum(ideal_indices != self.quantized_indices)
        symbol_error_rate = hamming_distance / len(original_signal)
        
        self.error_metrics = {
            'mse': mse,
            'hamming_distance': hamming_distance,
            'symbol_error_rate': symbol_error_rate
        }
        
        print("\n--- Error Analysis Results ---")
        print(f"Mean Squared Error (MSE):       {mse:.2e}")
        print(f"Hamming Distance (Symbol Errors): {hamming_distance} / {len(original_signal)}")
        print(f"Symbol Error Rate (SER):        {symbol_error_rate:.4%}")
        return self

    def compress_huffman(self):
        """
        Compresses the quantized indices using Huffman coding.
        Must be called after a quantization method.
        """
        if self.quantized_indices is None:
            raise RuntimeError("A quantization method must be applied before compression.")

        # 1. Calculate frequency of each quantization index
        frequency = Counter(self.quantized_indices)
        
        # 2. Build the Huffman tree using a priority queue (min-heap)
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # The Huffman codebook (dictionary of symbol -> code)
        huffman_codes = dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))

        # 3. Encode the signal indices
        encoded_signal = "".join([huffman_codes[s] for s in self.quantized_indices])

        # 4. Calculate compression statistics
        num_levels = self.quantization_params['num_levels']
        bits_per_symbol_fixed = math.ceil(math.log2(num_levels))
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
        """Applies Differential Pulse-Code Modulation (a predictive source coder)."""
        signal_to_code = self.get_active_signal()
        # Simple predictor: next sample will be like the last quantized sample
        predictions = np.roll(signal_to_code, 1)
        predictions[0] = 0 # Initial prediction
        
        errors = signal_to_code - predictions
        
        # Quantize the error signal, not the original signal
        error_quantizer = SignalAnalyzer(errors, self.fs)
        error_quantizer.quantize_uniform(num_levels)
        
        self.quantized_indices = error_quantizer.quantized_indices
        self.quantization_params = {'type': 'DPCM', 'num_levels': num_levels}
        
        # Reconstruct the signal to store as 'quantized_signal'
        reconstructed_errors = error_quantizer.quantized_signal
        reconstructed_signal = np.zeros_like(signal_to_code)
        for i in range(1, len(signal_to_code)):
            reconstructed_signal[i] = reconstructed_signal[i-1] + reconstructed_errors[i]
        self.quantized_signal = reconstructed_signal
        
        print(f"Applied DPCM with {num_levels} error quantization levels.")
        return self

    #

    def calculate_psd(self, nperseg: int = 2048):
        """
        Calculates Power Spectral Density (PSD) for the current signal and filtered signal.
        """
        def _get_psd(signal_data):
            freqs, Pxx = welch(signal_data, self.fs, nperseg=nperseg)
            Pxx_db = 10 * np.log10(Pxx)
            Pxx_db_norm = Pxx_db - np.max(Pxx_db)
            return freqs, Pxx_db_norm

        self.psd_results['original'] = _get_psd(self.signal)
        if self.filtered_signal is not None:
            self.psd_results['filtered'] = _get_psd(self.filtered_signal)
        print("Calculated Power Spectral Density.")
        return self

    def calculate_acf(self):
        """
        Calculates Autocorrelation Function (ACF) for the current signal and filtered signal.
        """
        def _get_acf(signal_data):
            demeaned = signal_data - np.mean(signal_data)
            acf = correlate(demeaned, demeaned, mode='full')
            acf /= np.max(acf)
            lags = np.arange(-(len(signal_data) - 1), len(signal_data)) / self.fs
            return lags, acf
            
        self.acf_results['original'] = _get_acf(self.signal)
        if self.filtered_signal is not None:
            self.acf_results['filtered'] = _get_acf(self.filtered_signal)
        print("Calculated Autocorrelation Function.")
        return self
