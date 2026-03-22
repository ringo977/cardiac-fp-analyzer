"""
plotting.py — Smart plotting for very long cardiac FP traces.

Uses min-max downsampling to render 360k-point traces while
preserving all visually important features (spikes, peaks).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def minmax_downsample(x, y, target_points=5000):
    """
    Min-max downsampling: for each bucket, keep both the min and max points.
    Guarantees all spikes and peaks are visible.
    """
    n = len(x)
    if n <= target_points: return x, y
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n_buckets = target_points // 2
    bucket_size = n / n_buckets
    out_x, out_y = [], []
    for i in range(n_buckets):
        start, end = int(i * bucket_size), min(int((i + 1) * bucket_size), n)
        seg_y = y[start:end]
        idx_min, idx_max = start + np.argmin(seg_y), start + np.argmax(seg_y)
        if idx_min < idx_max:
            out_x.extend([x[idx_min], x[idx_max]])
            out_y.extend([y[idx_min], y[idx_max]])
        else:
            out_x.extend([x[idx_max], x[idx_min]])
            out_y.extend([y[idx_max], y[idx_min]])
    return np.array(out_x), np.array(out_y)


def plot_raw_trace(df, metadata, channel='el1', target_points=8000,
                   save_path=None, figsize=(16, 4)):
    """Plot a single raw FP trace with smart downsampling."""
    fig, ax = plt.subplots(figsize=figsize)
    t, y = df['time'].values, df[channel].values
    t_ds, y_ds = minmax_downsample(t, y, target_points=target_points)
    ax.plot(t_ds, y_ds * 1000, linewidth=0.5, color='#1f77b4')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Voltage (mV)')
    ax.set_title(f"{metadata.get('filename', 'Unknown')} — {channel.upper()}", fontsize=10)
    ax.grid(True, alpha=0.3)
    fs = metadata.get('sample_rate', '?')
    n = metadata.get('n_samples', len(df))
    dur = n / fs if isinstance(fs, (int, float)) else '?'
    info_text = f"Fs={fs} Hz | N={n} | Duration={dur:.1f}s" if dur != '?' else f"Fs={fs} Hz | N={n}"
    ax.text(0.01, 0.97, info_text, transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return fig, ax


def plot_both_channels(df, metadata, target_points=8000, save_path=None, figsize=(16, 7)):
    """Plot both channels stacked vertically."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    t = df['time'].values
    for i, el in enumerate(['el1', 'el2']):
        y = df[el].values
        t_ds, y_ds = minmax_downsample(t, y, target_points=target_points)
        axes[i].plot(t_ds, y_ds * 1000, linewidth=0.5, color=['#1f77b4', '#d62728'][i])
        axes[i].set_ylabel(f'{el.upper()} (mV)'); axes[i].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (s)')
    axes[0].set_title(f"{metadata.get('filename', 'Unknown')} — Raw Traces", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return fig, axes


def plot_beat_overlay(beats_time, beats_data, metadata, channel='el1',
                      save_path=None, figsize=(10, 6)):
    """Overlay all detected beats aligned to depolarization spike."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (t, y) in enumerate(zip(beats_time, beats_data)):
        alpha = max(0.1, 1.0 - i * 0.01)
        ax.plot(t * 1000, y * 1000, linewidth=0.5, alpha=alpha, color='#1f77b4')
    ax.set_xlabel('Time from depolarization (ms)'); ax.set_ylabel('Voltage (mV)')
    ax.set_title(f"{metadata.get('filename', '')} — Beat Overlay ({len(beats_data)} beats)", fontsize=10)
    ax.grid(True, alpha=0.3); ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return fig, ax


def plot_analysis_summary(df_time, filtered, beat_indices, params, metadata,
                          channel='el1', save_path=None, figsize=(16, 14)):
    """
    4-panel analysis summary:
      1. Full filtered trace with beat markers
      2. Zoomed view of ~5 beats
      3. Beat period trend
      4. FPD trend
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, height_ratios=[1.5, 1.5, 1, 1], hspace=0.35)
    t = df_time.values if hasattr(df_time, 'values') else df_time
    y = filtered

    ax1 = fig.add_subplot(gs[0])
    t_ds, y_ds = minmax_downsample(t, y * 1000, target_points=8000)
    ax1.plot(t_ds, y_ds, linewidth=0.4, color='#1f77b4')
    if len(beat_indices) > 0:
        ax1.plot(t[beat_indices], y[beat_indices] * 1000, 'rv', markersize=3, alpha=0.6)
    ax1.set_ylabel('Voltage (mV)')
    el_label = metadata.get('analyzed_channel', '')
    title_suffix = f" [{el_label.upper()}]" if el_label else ""
    ax1.set_title(f"{metadata.get('filename', '')}{title_suffix} — Analysis Summary", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    if len(beat_indices) >= 3:
        mid = len(beat_indices) // 2
        fs_val = params.get('fs', 2000)
        start_idx = max(0, beat_indices[mid - 1] - int(0.1 * fs_val))
        n_show = min(5, len(beat_indices) - mid + 1)
        end_idx = min(len(t) - 1, beat_indices[min(mid + n_show, len(beat_indices) - 1)] + int(0.3 * fs_val))
        ax2.plot(t[start_idx:end_idx], y[start_idx:end_idx] * 1000, linewidth=0.8, color='#1f77b4')
        for bi in beat_indices:
            if start_idx <= bi <= end_idx:
                ax2.axvline(t[bi], color='red', alpha=0.3, linewidth=0.5)
    ax2.set_ylabel('Voltage (mV)'); ax2.set_xlabel('Time (s)')
    ax2.set_title('Zoomed View (~5 beats)', fontsize=9); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    bps = params.get('beat_periods', [])
    if len(bps) > 0:
        ax3.plot(range(1, len(bps) + 1), np.array(bps) * 1000, 'o-', markersize=3, linewidth=1, color='#2ca02c')
        mean_bp = np.mean(bps) * 1000
        ax3.axhline(mean_bp, color='gray', linestyle='--', alpha=0.5)
        ax3.text(0.98, 0.95, f'Mean BP = {mean_bp:.0f} ms\nBPM = {60000/mean_bp:.1f}',
                 transform=ax3.transAxes, fontsize=8, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax3.set_ylabel('Beat Period (ms)'); ax3.set_title('Beat Period (RR Interval) Trend', fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    fpds = params.get('fpd_values', [])
    fpdcs = params.get('fpdc_values', [])
    if len(fpds) > 0:
        ax4.plot(range(1, len(fpds) + 1), np.array(fpds) * 1000, 's-', markersize=3, linewidth=1, color='#d62728', label='FPD')
    if len(fpdcs) > 0:
        ax4.plot(range(1, len(fpdcs) + 1), np.array(fpdcs) * 1000, '^-', markersize=3, linewidth=1, color='#9467bd', label='FPDc (Fridericia)')
    if len(fpds) > 0:
        mean_fpd = np.mean(fpds) * 1000
        ax4.text(0.98, 0.95, f'Mean FPD = {mean_fpd:.0f} ms', transform=ax4.transAxes, fontsize=8, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax4.set_ylabel('FPD (ms)'); ax4.set_xlabel('Beat #')
    ax4.set_title('Field Potential Duration Trend', fontsize=9); ax4.grid(True, alpha=0.3)
    if len(fpds) > 0 and len(fpdcs) > 0: ax4.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return fig
