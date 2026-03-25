"""

Phase-Amplitude Coupling (PAC) analysis using the Modulation Index (MI)
method of Tort et al. (2010). Computes a comodulogram across a population
of cells and produces summary plots for dominant phase and amplitude
frequencies.

Reference:
    Tort ABL et al. (2010). Measuring phase-amplitude coupling between
    neuronal oscillations of different frequencies. J Neurophysiol,
    104(2), 1195-1210.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 3):
    """Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order (default 3).

    Returns
    -------
    b, a : array_like
        Filter coefficients.
    """
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 3,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1-D signal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
        Input signal.
    lowcut, highcut : float
        Passband edges (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order (default 3).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


def modulation_index(a_mean: np.ndarray) -> float:
    """Compute the Modulation Index (MI) from mean amplitude per phase bin.

    Normalises the amplitude distribution across phase bins and returns
    1 - normalised entropy (Tort et al., 2010).

    Parameters
    ----------
    a_mean : np.ndarray, shape (n_bins,)
        Mean amplitude in each phase bin.

    Returns
    -------
    float
        MI value in [0, 1]; 0 = uniform, 1 = fully concentrated.
    """
    P = a_mean / np.sum(a_mean)
    H = -np.sum(P * np.log(P + 1e-12))
    H_max = np.log(len(P))
    return (H_max - H) / H_max


def compute_pac(
    signal_data: np.ndarray,
    phase_band: tuple,
    amp_band: tuple,
    fs: float,
    n_bins: int = 18,
) -> float:
    """Compute PAC between two frequency bands for a single signal.

    Extracts the instantaneous phase of ``phase_band`` and the envelope
    of ``amp_band`` via the Hilbert transform, then bins the envelope by
    phase and evaluates the MI.

    Parameters
    ----------
    signal_data : np.ndarray, shape (n_samples,)
        Input time series.
    phase_band : tuple of (float, float)
        (low, high) cutoff frequencies for the phase signal (Hz).
    amp_band : tuple of (float, float)
        (low, high) cutoff frequencies for the amplitude signal (Hz).
    fs : float
        Sampling frequency (Hz).
    n_bins : int
        Number of phase bins spanning [-π, π] (default 18).

    Returns
    -------
    float
        Modulation Index for this band pair.
    """
    low_filtered = butter_bandpass_filter(signal_data, *phase_band, fs)
    high_filtered = butter_bandpass_filter(signal_data, *amp_band, fs)

    phase = np.angle(hilbert(low_filtered))
    amp = np.abs(hilbert(high_filtered))

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    a_mean = np.zeros(n_bins)

    for k in range(n_bins):
        mask = (phase >= bins[k]) & (phase < bins[k + 1])
        if np.any(mask):
            a_mean[k] = np.mean(amp[mask])

    return modulation_index(a_mean)


def compute_comodulogram(
    data: np.ndarray,
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    fs: float,
    phase_bw: float = 0.01,
    amp_bw: float = 0.1,
    n_bins: int = 18,
) -> np.ndarray:
    """Compute the full PAC comodulogram for a population of cells.

    Parameters
    ----------
    data : np.ndarray, shape (n_cells, n_samples)
        Normalised fluorescence (or other) traces, one row per cell.
    phase_freqs : np.ndarray
        Centre frequencies for the phase axis (Hz).
    amp_freqs : np.ndarray
        Centre frequencies for the amplitude axis (Hz).
    fs : float
        Sampling frequency (Hz).
    phase_bw : float
        Half-bandwidth for phase bands (Hz); band = [f, f + phase_bw].
    amp_bw : float
        Half-bandwidth for amplitude bands (Hz).
    n_bins : int
        Number of phase bins (default 18).

    Returns
    -------
    comod_all : np.ndarray, shape (n_cells, n_amp_freqs, n_phase_freqs)
        Per-cell MI values across the frequency grid.
    """
    n_cells = data.shape[0]
    comod_all = np.zeros((n_cells, len(amp_freqs), len(phase_freqs)))

    for c in range(n_cells):
        print(f"Processing cell {c + 1}/{n_cells}")
        for i, pf in enumerate(phase_freqs):
            for j, af in enumerate(amp_freqs):
                comod_all[c, j, i] = compute_pac(
                    data[c],
                    phase_band=(pf, pf + phase_bw),
                    amp_band=(af, af + amp_bw),
                    fs=fs,
                    n_bins=n_bins,
                )

    return comod_all


def plot_comodulogram(
    comod_mean: np.ndarray,
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    phase_bw: float,
    amp_bw: float,
) -> None:
    """Plot the mean PAC comodulogram.

    Parameters
    ----------
    comod_mean : np.ndarray, shape (n_amp_freqs, n_phase_freqs)
        Mean MI across cells.
    phase_freqs, amp_freqs : np.ndarray
        Frequency axes (Hz).
    phase_bw, amp_bw : float
        Bandwidths used when computing the comodulogram (Hz).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        comod_mean,
        aspect="auto",
        origin="lower",
        extent=[
            phase_freqs[0],
            phase_freqs[-1] + phase_bw,
            amp_freqs[0],
            amp_freqs[-1] + amp_bw,
        ],
        cmap="jet",
    )
    fig.colorbar(im, ax=ax, label="Modulation Index")
    ax.set_xlabel("Phase frequency (Hz)")
    ax.set_ylabel("Amplitude frequency (Hz)")
    ax.set_title("PAC Comodulogram (mean across cells)")
    plt.tight_layout()
    plt.show()


def plot_dominant_frequencies(
    comod_all: np.ndarray,
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
) -> None:
    """Plot marginal MI profiles for phase and amplitude axes.

    Parameters
    ----------
    comod_all : np.ndarray, shape (n_cells, n_amp_freqs, n_phase_freqs)
        Per-cell comodulogram.
    phase_freqs, amp_freqs : np.ndarray
        Frequency axes (Hz).
    """
    n_cells = comod_all.shape[0]

    MI_phase = np.mean(comod_all, axis=1)
    MI_phase_mean = np.mean(MI_phase, axis=0)
    MI_phase_sem = np.std(MI_phase, axis=0) / np.sqrt(n_cells)

    MI_amp = np.mean(comod_all, axis=2)
    MI_amp_mean = np.mean(MI_amp, axis=0)
    MI_amp_sem = np.std(MI_amp, axis=0) / np.sqrt(n_cells)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(phase_freqs, MI_phase_mean, yerr=MI_phase_sem, fmt="-o", capsize=3)
    axes[0].set_xlabel("Phase frequency (Hz)")
    axes[0].set_ylabel("Modulation Index")
    axes[0].set_title("Dominant phase frequencies")
    axes[0].grid(True)

    axes[1].errorbar(amp_freqs, MI_amp_mean, yerr=MI_amp_sem, fmt="-o", capsize=3)
    axes[1].set_xlabel("Amplitude frequency (Hz)")
    axes[1].set_ylabel("Modulation Index")
    axes[1].set_title("Dominant amplitude frequencies")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Frequency grid
    phase_freqs = np.arange(0.005, 0.05, 0.005)
    amp_freqs = np.arange(0.05, 0.8, 0.05)
    phase_bw = 0.01
    amp_bw = 0.10

    # Compute per-cell comodulograms
    # `datosNorm_exponential` and `fs` must be defined in the calling scope
    comod_all = compute_comodulogram(
        data=datosNorm_exponential,
        phase_freqs=phase_freqs,
        amp_freqs=amp_freqs,
        fs=fs,
        phase_bw=phase_bw,
        amp_bw=amp_bw,
    )

    # Mean comodulogram
    comod_mean = np.mean(comod_all, axis=0)

    plot_comodulogram(comod_mean, phase_freqs, amp_freqs, phase_bw, amp_bw)
    plot_dominant_frequencies(comod_all, phase_freqs, amp_freqs)