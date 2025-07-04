"""
Advanced Signal Processing, Communications, and Compression Class (PyTorch Version)

This script defines a generic SignalAnalyzer class for filtering, analog/digital modulation,
quantization, compression, and analysis of 1D signal data using PyTorch.

NOTE: This version uses PyTorch for tensor operations to enable potential GPU acceleration.
However, it retains dependencies on SciPy and Scikit-Learn for specialized functions
(e.g., filter design, Welch's method, Hilbert transform) that do not have
native, stable equivalents in PyTorch. The class handles the conversion between
PyTorch Tensors and NumPy arrays where necessary.
"""

import math
import heapq
from collections import Counter
from typing import Union, List, Optional, Tuple, Dict

import torch
import numpy as np
from scipy.signal import welch, correlate, butter, filtfilt, hilbert, gaussian
from sklearn.cluster import KMeans


class SignalAnalyzer:
    """
    A generic PyTorch-based class for signal processing steps for any 1D signal.

    This class provides a comprehensive toolkit for analyzing and manipulating
    1D signals. It supports a chained-method interface for a fluid workflow,
    covering operations from basic filtering to advanced digital modulation and
    source coding, with the ability to leverage GPU acceleration for tensor operations.
    """

    def __init__(self,
                 signal_data: Union[np.ndarray, List[float], torch.Tensor],
                 fs: float,
                 device: Optional[str] = None):
        """
        Initializes the analyzer with signal data and its sampling rate.

        Args:
            signal_data (Union[np.ndarray, List[float], torch.Tensor]): The 1D
                array-like object containing the signal data.
            fs (float): The sampling frequency of the signal in Hz.
            device (Optional[str]): The PyTorch device to use ('cuda', 'cpu').
                If None, it will auto-detect CUDA availability.
        """
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise ValueError("fs (sampling frequency) must be a positive number.")

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert input to a float64 tensor on the specified device for precision
        self.full_signal = torch.as_tensor(signal_data, dtype=torch.float64, device=self.device)
        if self.full_signal.ndim != 1:
            raise TypeError("signal_data must be a 1D array-like object.")

        self.fs = fs
        self.full_times = torch.arange(len(self.full_signal), device=self.device, dtype=torch.float64) / self.fs

        # Initialize to analyze the entire signal by default
        self.signal = self.full_signal
        self.times = self.full_times
        self._reset_downstream_analyses()

        print(f"SignalAnalyzer initialized successfully on device: '{self.device}'.")

    def _reset_downstream_analyses(self):
        """Resets all calculated results that depend on the current signal or filter state."""
        self.filtered_signal: Optional[torch.Tensor] = None
        self.quantized_signal: Optional[torch.Tensor] = None
        self.quantized_indices: Optional[torch.Tensor] = None
        self.kmeans_model: Optional[KMeans] = None
        self.modulated_signal: Optional[torch.Tensor] = None
        self.demodulated_signal: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        self.bitstream: Optional[str] = None
        self.baseband_i: Optional[torch.Tensor] = None
        self.baseband_q: Optional[torch.Tensor] = None

        self.filter_params: Dict = {}
        self.quantization_params: Dict = {}
        self.modulation_params: Dict = {}
        self.pulse_shaping_params: Dict = {}
        self.compression_results: Dict = {}
        self.error_metrics: Dict = {}
        self.histogram_results: Dict = {}
        self.psd_results: Dict = {}
        self.acf_results: Dict = {}

    def get_active_signal(self) -> torch.Tensor:
        """Returns the current signal being analyzed (filtered or original)."""
        return self.filtered_signal if self.filtered_signal is not None else self.signal

    def select_time_interval(self, start_time: float, end_time: float):
        """Focuses all subsequent analysis on a sub-segment of the signal."""
        start_sample = max(0, int(start_time * self.fs))
        end_sample = min(len(self.full_signal), int(end_time * self.fs))

        if start_sample >= end_sample:
            raise ValueError("Invalid time interval. Start must be before end and within bounds.")

        self.signal = self.full_signal[start_sample:end_sample]
        self.times = self.full_times[start_sample:end_sample]
        self._reset_downstream_analyses()

        print(f"Analysis focused on time interval: {self.times[0]:.2f}s to {self.times[-1]:.2f}s")
        return self

    # --- FILTERING METHODS (Using SciPy Bridge) ---

    def _apply_filter(self, b: np.ndarray, a: np.ndarray):
        """Internal helper to apply a SciPy filter to a PyTorch tensor."""
        self._reset_downstream_analyses()
        signal_np = self.signal.cpu().numpy()
        filtered_np = filtfilt(b, a, signal_np)
        self.filtered_signal = torch.as_tensor(filtered_np, dtype=torch.float64, device=self.device)

    def apply_lowpass_filter(self, cutoff: float, order: int = 4):
        """Applies a Butterworth low-pass filter."""
        b, a = butter(order, cutoff / (0.5 * self.fs), btype='low')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'lowpass', 'cutoff': cutoff, 'order': order}
        print(f"Applied a low-pass filter at {cutoff} Hz.")
        return self

    def apply_highpass_filter(self, cutoff: float, order: int = 4):
        """Applies a Butterworth high-pass filter."""
        b, a = butter(order, cutoff / (0.5 * self.fs), btype='high')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'highpass', 'cutoff': cutoff, 'order': order}
        print(f"Applied a high-pass filter at {cutoff} Hz.")
        return self

    def apply_bandpass_filter(self, lowcut: float, highcut: float, order: int = 4):
        """Applies a Butterworth band-pass filter."""
        b, a = butter(order, [lowcut / (0.5 * self.fs), highcut / (0.5 * self.fs)], btype='bandpass')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'bandpass', 'lowcut': lowcut, 'highcut': highcut, 'order': order}
        print(f"Applied a band-pass filter from {lowcut} to {highcut} Hz.")
        return self

    def apply_bandstop_filter(self, lowcut: float, highcut: float, order: int = 4):
        """Applies a Butterworth band-stop filter."""
        b, a = butter(order, [lowcut / (0.5 * self.fs), highcut / (0.5 * self.fs)], btype='bandstop')
        self._apply_filter(b, a)
        self.filter_params = {'type': 'bandstop', 'lowcut': lowcut, 'highcut': highcut, 'order': order}
        print(f"Applied a band-stop filter from {lowcut} to {highcut} Hz.")
        return self

    # --- ANALOG MODULATION / DEMODULATION METHODS ---

    def _generate_carrier(self, carrier_freq: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates cosine and sine carrier waves as PyTorch tensors."""
        t = 2 * math.pi * carrier_freq * self.times
        return torch.cos(t), torch.sin(t)

    def modulate_am(self, carrier_freq: float, modulation_index: float = 0.5):
        """Modulates the active signal using Amplitude Modulation (AM)."""
        m_t = self.get_active_signal()
        m_t_norm = m_t / torch.max(torch.abs(m_t))
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
        
        m_t_np = m_t.cpu().numpy()
        m_h_np = np.imag(hilbert(m_t_np))
        m_h = torch.as_tensor(m_h_np, dtype=torch.float64, device=self.device)

        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        self.modulated_signal = m_t * carrier_cos - m_h * carrier_sin if side == 'upper' else m_t * carrier_cos + m_h * carrier_sin
        self.modulation_params = {'type': 'SSB-SC', 'carrier_freq': carrier_freq, 'side': side}
        print(f"Applied {side}-sideband SSB-SC with fc={carrier_freq} Hz.")
        return self

    def modulate_pm(self, carrier_freq: float, deviation: float = math.pi/2):
        """Modulates using Phase Modulation (PM)."""
        m_t = self.get_active_signal()
        m_t_norm = m_t / torch.max(torch.abs(m_t))
        carrier_phase = 2 * math.pi * carrier_freq * self.times
        self.modulated_signal = torch.cos(carrier_phase + deviation * m_t_norm)
        self.modulation_params = {'type': 'PM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied PM with fc={carrier_freq} Hz.")
        return self

    def modulate_fm(self, carrier_freq: float, deviation: float = 50.0):
        """Modulates using Frequency Modulation (FM)."""
        m_t = self.get_active_signal()
        m_t_norm = m_t / torch.max(torch.abs(m_t))
        phase_dev = 2 * math.pi * deviation * torch.cumsum(m_t_norm, dim=0) / self.fs
        carrier_phase = 2 * math.pi * carrier_freq * self.times
        self.modulated_signal = torch.cos(carrier_phase + phase_dev)
        self.modulation_params = {'type': 'FM', 'carrier_freq': carrier_freq, 'deviation': deviation}
        print(f"Applied FM with fc={carrier_freq} Hz.")
        return self
        
    def demodulate_coherent(self, lpf_cutoff: float = None):
        """Demodulates AM, DSB-SC, or SSB-SC using a coherent receiver."""
        if self.modulated_signal is None: raise RuntimeError("No modulated signal to demodulate.")
        s_t, fc = self.modulated_signal, self.modulation_params['carrier_freq']
        carrier_cos, _ = self._generate_carrier(fc)
        v_t = s_t * carrier_cos

        lpf_cutoff = lpf_cutoff or fc * 0.5
        b, a = butter(5, lpf_cutoff / (0.5 * self.fs), btype='low')
        demod_np = filtfilt(b, a, v_t.cpu().numpy())
        self.demodulated_signal = torch.as_tensor(demod_np, dtype=torch.float64, device=self.device)
        print(f"Applied coherent demodulation with LPF at {lpf_cutoff} Hz.")
        return self

    def demodulate_angle(self):
        """Demodulates PM or FM signals using a phase-based method."""
        if self.modulated_signal is None: raise RuntimeError("No modulated signal.")
        mod_type = self.modulation_params.get('type')
        if mod_type not in ['PM', 'FM']: raise RuntimeError("Angle demodulation only for PM/FM.")
        
        s_t, fc = self.modulated_signal, self.modulation_params['carrier_freq']
        
        s_t_np = s_t.cpu().numpy()
        analytic_signal = hilbert(s_t_np)
        phase_np = np.unwrap(np.angle(analytic_signal))
        instantaneous_phase = torch.from_numpy(phase_np).to(dtype=torch.float64, device=self.device)

        if mod_type == 'PM':
            self.demodulated_signal = instantaneous_phase - (2 * math.pi * fc * self.times)
        else: # FM
            inst_freq = torch.diff(instantaneous_phase) / (2 * math.pi / self.fs)
            self.demodulated_signal = torch.cat((inst_freq, inst_freq[-1:])) - fc

        print(f"Applied angle demodulation for {mod_type}.")
        return self

    # --- QUANTIZATION METHODS ---

    def quantize_uniform(self, num_levels: int):
        """Quantizes the active signal using uniform step sizes."""
        signal = self.get_active_signal()
        min_val, max_val = torch.min(signal), torch.max(signal)

        if max_val == min_val:
            self.quantized_signal = torch.full_like(signal, min_val)
            self.quantized_indices = torch.zeros_like(signal, dtype=torch.long)
        else:
            step_size = (max_val - min_val) / num_levels
            indices = torch.floor((signal - min_val) / step_size)
            self.quantized_indices = torch.clip(indices, 0, num_levels - 1).to(torch.long)
            self.quantized_signal = min_val + self.quantized_indices * step_size + (step_size / 2)

        self.quantization_params = {'type': 'uniform', 'num_levels': num_levels, 'min_val': min_val.item(), 'max_val': max_val.item()}
        print(f"Applied uniform quantization with {num_levels} levels.")
        return self
        
    def quantize_non_uniform_mu_law(self, num_levels: int, mu: int = 255):
        """Quantizes the active signal using μ-law companding."""
        signal = self.get_active_signal()
        max_abs_val = torch.max(torch.abs(signal))
        if max_abs_val == 0:
            self.quantized_signal = torch.zeros_like(signal)
            self.quantized_indices = torch.zeros_like(signal, dtype=torch.long)
            return self

        norm_signal = signal / max_abs_val
        comp_signal = torch.sign(norm_signal) * torch.log(1 + mu * torch.abs(norm_signal)) / math.log(1 + mu)
        
        step_comp = 2.0 / num_levels
        indices = torch.floor((comp_signal + 1) / step_comp)
        self.quantized_indices = torch.clip(indices, 0, num_levels - 1).to(torch.long)
        
        recon_comp = -1 + self.quantized_indices * step_comp + (step_comp / 2)
        exp_signal = torch.sign(recon_comp) * ((1 + mu)**torch.abs(recon_comp) - 1) / mu
        self.quantized_signal = exp_signal * max_abs_val

        self.quantization_params = {'type': 'mu-law', 'num_levels': num_levels, 'mu': mu, 'max_abs_val': max_abs_val.item()}
        print(f"Applied μ-law quantization with {num_levels} levels.")
        return self

    def quantize_non_uniform_kmeans(self, num_levels: int):
        """Quantizes the active signal using K-Means clustering."""
        signal_np = self.get_active_signal().reshape(-1, 1).cpu().numpy()
        kmeans = KMeans(n_clusters=num_levels, n_init='auto', random_state=42).fit(signal_np)
        self.kmeans_model = kmeans
        
        levels_np, indices_np = kmeans.cluster_centers_.flatten(), kmeans.labels_
        levels = torch.from_numpy(levels_np).to(dtype=torch.float64, device=self.device)
        indices = torch.from_numpy(indices_np).to(dtype=torch.long, device=self.device)
        
        sorted_level_indices = torch.argsort(levels)
        sorted_levels = levels[sorted_level_indices]
        
        remap_array = torch.zeros_like(sorted_level_indices); remap_array[sorted_level_indices] = torch.arange(num_levels, device=self.device)
        
        self.quantized_indices = remap_array[indices]
        self.quantized_signal = sorted_levels[self.quantized_indices]
        self.quantization_params = {'type': 'kmeans', 'num_levels': num_levels, 'levels': sorted_levels}
        print(f"Applied K-Means quantization with {num_levels} levels.")
        return self

    # --- DIGITAL DATA GENERATION AND PULSE SHAPING ---

    def generate_bitstream_from_indices(self):
        """Converts quantized indices into a single binary string (bitstream)."""
        if self.quantized_indices is None: raise RuntimeError("Quantization must be applied first.")
        num_levels = self.quantization_params.get('num_levels')
        if num_levels is None: raise RuntimeError("Quantization parameters not found.")
        
        bits_per_symbol = math.ceil(math.log2(num_levels))
        indices_list = self.quantized_indices.cpu().tolist()
        self.bitstream = "".join([np.binary_repr(idx, width=bits_per_symbol) for idx in indices_list])
        print(f"Generated bitstream with {len(self.bitstream)} bits ({bits_per_symbol} bits/symbol).")
        return self

    def _generate_baseband_from_bits(self, samples_per_symbol: int, bits_per_symbol: int = 1) -> torch.Tensor:
        """Helper to create a baseband signal from the bitstream by mapping bits to symbols."""
        if self.bitstream is None: self.generate_bitstream_from_indices()
        
        rem = len(self.bitstream) % bits_per_symbol
        bits_padded = self.bitstream + '0' * ((bits_per_symbol - rem) % bits_per_symbol)
        
        symbols_list = [int(bits_padded[i:i+bits_per_symbol], 2) for i in range(0, len(bits_padded), bits_per_symbol)]
        symbols = torch.tensor(symbols_list, dtype=torch.long, device=self.device)

        self.baseband_i = symbols.repeat_interleave(samples_per_symbol).to(torch.float64)
        self.baseband_q = None
        return symbols
        
    def apply_gaussian_pulse_shaping(self, samples_per_symbol: int, bt_product: float = 0.5):
        """Applies a Gaussian filter to the I and Q baseband signals using PyTorch convolution."""
        if self.baseband_i is None: raise RuntimeError("Baseband signal must be generated first.")
        
        t_np = np.arange(-samples_per_symbol * 2, samples_per_symbol * 2 + 1)
        std_dev = samples_per_symbol / (math.sqrt(2 * math.log(2)) * (1/bt_product))
        gauss_filter_np = gaussian(len(t_np), std=std_dev)
        gauss_filter_np /= np.sum(gauss_filter_np)
        
        kernel = torch.as_tensor(gauss_filter_np, dtype=torch.float64, device=self.device).view(1, 1, -1)

        def shape_signal(signal_in: torch.Tensor) -> torch.Tensor:
            signal_in = signal_in.view(1, 1, -1)
            return torch.nn.functional.conv1d(signal_in, kernel, padding='same').view(-1)

        self.baseband_i = shape_signal(self.baseband_i)
        if self.baseband_q is not None: self.baseband_q = shape_signal(self.baseband_q)
            
        self.pulse_shaping_params = {'type': 'gaussian', 'BT': bt_product}
        print(f"Applied Gaussian pulse shaping with BT={bt_product}.")
        return self

    # --- DIGITAL MODULATION METHODS ---

    def modulate_ask(self, carrier_freq: float, samples_per_symbol: int):
        """Amplitude Shift Keying (On-Off Keying for binary)."""
        self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        carrier_cos, _ = self._generate_carrier(carrier_freq)
        self.modulated_signal = self.baseband_i * carrier_cos[:len(self.baseband_i)]
        self.modulation_params = {'type': 'ASK', 'carrier_freq': carrier_freq, 'M': 2}
        print(f"Applied ASK (OOK) modulation.")
        return self

    def modulate_psk(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """Phase Shift Keying (BPSK, QPSK, 8-PSK, etc.)."""
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        
        phases = 2 * math.pi * symbols / M
        self.baseband_i = torch.cos(phases).repeat_interleave(samples_per_symbol)
        self.baseband_q = torch.sin(phases).repeat_interleave(samples_per_symbol)
        
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        
        mod_type = {2: 'BPSK', 4: 'QPSK'}.get(M, f'{M}-PSK')
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_oqpsk(self, carrier_freq: float, samples_per_symbol: int):
        """Offset QPSK."""
        self.modulate_psk(carrier_freq, samples_per_symbol, M=4)
        offset = samples_per_symbol // 2
        self.baseband_q = torch.roll(self.baseband_q, shifts=offset, dims=0)
        
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params['type'] = 'OQPSK'
        print("Applied OQPSK modulation.")
        return self

    def modulate_fsk(self, carrier_freq: float, samples_per_symbol: int, M: int, freq_sep: float):
        """M-ary Frequency Shift Keying."""
        bits_per_symbol = int(np.log2(M))
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        
        freq_offsets = torch.arange(-(M-1)/2, (M-1)/2 + 1, device=self.device) * freq_sep
        freqs = carrier_freq + freq_offsets
        
        inst_freq = freqs[symbols].repeat_interleave(samples_per_symbol)
        phase = 2 * math.pi * torch.cumsum(inst_freq, dim=0) / self.fs
        self.modulated_signal = torch.cos(phase)
        
        mod_type = 'BFSK' if M == 2 else f'{M}-FSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'M': M}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_cpm(self, carrier_freq: float, samples_per_symbol: int, h: float, L: int=1):
        """General Continuous Phase Modulation engine (for CPFSK, MSK)."""
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        alpha = 2 * symbols - 1 # Bipolar {-1, 1}
        
        g_t = torch.ones(L * samples_per_symbol, device=self.device, dtype=torch.float64) / (2 * L * samples_per_symbol)
        phase_pulses = torch.nn.functional.conv1d(
            alpha.repeat_interleave(samples_per_symbol).view(1, 1, -1),
            g_t.view(1, 1, -1), padding='full').view(-1)
        
        phase = math.pi * h * torch.cumsum(phase_pulses, dim=0) / self.fs
        t = torch.arange(len(phase), device=self.device, dtype=torch.float64) / self.fs
        carrier_phase = 2 * math.pi * carrier_freq * t
        self.modulated_signal = torch.cos(carrier_phase + phase)
        
        mod_type = 'MSK' if h == 0.5 and L == 1 else 'CPFSK'
        self.modulation_params = {'type': mod_type, 'carrier_freq': carrier_freq, 'h': h}
        print(f"Applied {mod_type} modulation.")
        return self

    def modulate_gmsk(self, carrier_freq: float, samples_per_symbol: int, bt_product: float = 0.3):
        """Gaussian Minimum Shift Keying."""
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol=1)
        alpha = 2 * symbols - 1
        
        shaper = SignalAnalyzer(alpha.repeat_interleave(samples_per_symbol), self.fs, device=str(self.device))
        shaper.baseband_i = shaper.signal
        shaper.apply_gaussian_pulse_shaping(samples_per_symbol, bt_product)
        freq_pulses = shaper.baseband_i

        phase = math.pi * 0.5 * torch.cumsum(freq_pulses, dim=0) / self.fs
        t = torch.arange(len(phase), device=self.device, dtype=torch.float64) / self.fs
        self.modulated_signal = torch.cos(2 * math.pi * carrier_freq * t + phase)
        self.modulation_params = {'type': 'GMSK', 'carrier_freq': carrier_freq, 'BT': bt_product}
        print("Applied GMSK modulation.")
        return self

    def modulate_qam(self, carrier_freq: float, samples_per_symbol: int, M: int):
        """M-ary Quadrature Amplitude Modulation."""
        bits_per_symbol = int(np.log2(M))
        if bits_per_symbol % 2 != 0: raise ValueError("For square QAM, M must be a power of 4 (4, 16, 64...).")
        
        symbols = self._generate_baseband_from_bits(samples_per_symbol, bits_per_symbol)
        sqrt_M = int(np.sqrt(M))
        i_levels = (2 * (symbols % sqrt_M) - (sqrt_M - 1))
        q_levels = (2 * (symbols // sqrt_M) - (sqrt_M - 1))

        self.baseband_i = i_levels.to(torch.float64).repeat_interleave(samples_per_symbol)
        self.baseband_q = q_levels.to(torch.float64).repeat_interleave(samples_per_symbol)
        
        carrier_cos, carrier_sin = self._generate_carrier(carrier_freq)
        len_match = len(self.baseband_i)
        self.modulated_signal = self.baseband_i * carrier_cos[:len_match] - self.baseband_q * carrier_sin[:len_match]
        self.modulation_params = {'type': f'{M}-QAM', 'carrier_freq': carrier_freq, 'M': M}
        print(f"Applied {M}-QAM modulation.")
        return self

    # --- DIGITAL DEMODULATION ---
    def demodulate_coherent_digital(self, lpf_cutoff: float = None):
        """Coherent demodulator for ASK, PSK, QAM."""
        s_t, fc = self.modulated_signal, self.modulation_params['carrier_freq']
        carrier_cos, carrier_sin = self._generate_carrier(fc)
        len_match = len(s_t)
        
        i_branch = s_t * carrier_cos[:len_match] * 2
        q_branch = s_t * (-carrier_sin[:len_match]) * 2

        lpf_cutoff = lpf_cutoff or fc * 0.9
        b, a = butter(5, lpf_cutoff / (0.5 * self.fs), btype='low')
        i_demod_np = filtfilt(b, a, i_branch.cpu().numpy())
        q_demod_np = filtfilt(b, a, q_branch.cpu().numpy())
        
        self.demodulated_signal = {
            'I': torch.as_tensor(i_demod_np, dtype=torch.float64, device=self.device),
            'Q': torch.as_tensor(q_demod_np, dtype=torch.float64, device=self.device)
        }
        print("Applied coherent digital demodulation.")
        return self

    # --- ERROR AND COMPRESSION METHODS ---

    def calculate_error_metrics(self):
        """Calculates error between the original active signal and its quantized version."""
        if self.quantized_signal is None: raise RuntimeError("Quantization must be applied first.")
        original_signal = self.get_active_signal()
        quant_params = self.quantization_params
        
        mse = torch.mean((original_signal - self.quantized_signal)**2)
        
        # ... logic to calculate ideal_indices using torch functions ...
        
        symbol_error_count = torch.sum(ideal_indices != self.quantized_indices)
        symbol_error_rate = symbol_error_count / len(original_signal)
        
        self.error_metrics = {'mse': mse.item(), 'symbol_error_count': symbol_error_count.item(), 'symbol_error_rate': symbol_error_rate.item()}
        print(f"\n--- Quantization Error Analysis ---\nMSE: {mse.item():.2e}\nSymbol Errors: {symbol_error_count.item()}/{len(original_signal)}\nSER: {symbol_error_rate.item():.4%}")
        return self

    def compress_huffman(self):
        """Compresses the quantized indices using Huffman coding."""
        if self.quantized_indices is None: raise RuntimeError("Quantization must be applied first.")
        
        freq = Counter(self.quantized_indices.cpu().numpy())
        if not freq: return self
        
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo, hi = heapq.heappop(heap), heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        huffman_codes = dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))
        encoded_signal = "".join([huffman_codes[s] for s in self.quantized_indices.cpu().numpy()])
        
        num_levels = self.quantization_params['num_levels']
        bits_fixed = math.ceil(math.log2(num_levels)) if num_levels > 1 else 1
        original_bits = len(self.quantized_indices) * bits_fixed
        compressed_bits = len(encoded_signal)
        ratio = original_bits / compressed_bits if compressed_bits > 0 else float('inf')

        self.compression_results = {'codes': huffman_codes, 'original_bits': original_bits, 'compressed_bits': compressed_bits, 'compression_ratio': ratio}
        print(f"\n--- Huffman Compression ---\nOriginal: {original_bits} bits\nCompressed: {compressed_bits} bits\nRatio: {ratio:.2f}:1")
        return self

    def apply_dpcm(self, num_levels: int):
        """Applies Differential Pulse-Code Modulation (DPCM)."""
        signal = self.get_active_signal()
        predictions = torch.roll(signal, shifts=1, dims=0)
        predictions[0] = 0
        errors = signal - predictions
        
        error_quantizer = SignalAnalyzer(errors, self.fs, device=str(self.device))
        error_quantizer.quantize_uniform(num_levels)
        
        self.quantized_indices = error_quantizer.quantized_indices
        self.quantization_params = {'type': 'DPCM', 'num_levels': num_levels}
        
        reconstructed = torch.zeros_like(signal)
        for i in range(1, len(signal)):
            prediction = reconstructed[i-1]
            reconstructed[i] = prediction + error_quantizer.quantized_signal[i]
        self.quantized_signal = reconstructed
        
        print(f"Applied DPCM with {num_levels} error quantization levels.")
        return self

    # --- ANALYSIS METHODS (Using SciPy/NumPy Bridge) ---

    def calculate_psd(self, nperseg: int = 2048):
        """Calculates Power Spectral Density (PSD) using Welch's method."""
        def _get_psd(data: torch.Tensor): return welch(data.cpu().numpy(), self.fs, nperseg=nperseg)
        self.psd_results['original'] = _get_psd(self.signal)
        if self.filtered_signal is not None: self.psd_results['filtered'] = _get_psd(self.filtered_signal)
        print("Calculated Power Spectral Density.")
        return self

    def calculate_acf(self):
        """Calculates the Autocorrelation Function (ACF)."""
        def _get_acf(data: torch.Tensor):
            data_np = data.cpu().numpy()
            demeaned = data_np - np.mean(data_np)
            acf = correlate(demeaned, demeaned, mode='full')
            acf /= acf[len(demeaned) - 1] # Normalize
            lags = np.arange(-(len(data_np) - 1), len(data_np)) / self.fs
            return lags, acf
        self.acf_results['original'] = _get_acf(self.signal)
        if self.filtered_signal is not None: self.acf_results['filtered'] = _get_acf(self.filtered_signal)
        print("Calculated Autocorrelation Function.")
        return self
        
    def calculate_histogram(self, bins: int = 100):
        """Calculates the amplitude distribution histogram."""
        signal_np = self.get_active_signal().cpu().numpy()
        counts, bin_edges = np.histogram(signal_np, bins=bins, density=True)
        self.histogram_results = {'counts': counts, 'bin_edges': bin_edges, 'bins': bins}
        print(f"Calculated histogram with {bins} bins.")
        return self
