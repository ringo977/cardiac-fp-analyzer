#!/usr/bin/env python
"""Build the real-signal regression corpus under tests/fixtures/real_signals/.

For each baseline recording with a published reference value (Visone et al.
2023 Excel sheets, extracted into data_reference/ground_truth.json), store a
60-second window of the electrode the authors used, at full sampling rate,
together with the reference RR / FPD. The test suite
(tests/test_real_signal_regression.py) rebuilds a WaveForms-style CSV from
each fixture and runs the full pipeline on it.

Why a window and not the whole file: the source CSVs are ~22 MB each; a
60 s float32 window is ~0.5 MB and compresses further. Why full rate: the
beat detector's refractory and derivative thresholds are rate-dependent and
must be exercised as in production.

Usage (from repo root, with the dataset mounted):
    python tools/build_real_signal_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardiac_fp_analyzer.loader import load_csv  # noqa: E402

DATASET = ROOT / 'Experiments-Toxicol.Science'
OUT = ROOT / 'tests' / 'fixtures' / 'real_signals'

T0_S = 30.0     # skip the first 30 s (settling / trigger transients)
DUR_S = 90.0    # 60 s windows proved too short for a stable RR median on irregular rhythms
# Per-label overrides: (t0_s, duration_s). Exp8 chipD has CV(RR) ~32 %, so only
# the full recording reproduces the published RR; it is also the motivating case.
WINDOW_OVERRIDE = {'exp8_d6_chipD_ch1': (0.0, 180.0)}

# (relative path, authors' electrode, RR_ms, FPD_ms, label, notes)
# Electrode side from data_reference/sides_resolved.json (sx=el1, dx=el2),
# confirmed for Exp8 chipD by the paired FPD comparison in HANDOFF (+1 %).
CORPUS = [
    ('Exp8/Day6/chipD_ch1_baseline.csv', 'el1', 2115.6, 609.1, 'exp8_d6_chipD_ch1',
     'Motivating case for the noise-floor gate: 156 detections vs ~75 true '
     'beats before the fix (one noise-level insertion per RR interval). '
     'Dofetilide baseline. Full 180 s stored: CV(RR) ~32 %, shorter windows '
     'do not reproduce the published RR.'),
    ('Exp8/Day7/chipA_ch1_baseline.csv', 'el1', 2513.0, 835.0, 'exp8_d7_chipA_ch1',
     'Slow regular rhythm, positive polarity.'),
    ('Exp8/Day7/chipE_ch2_baseline.csv', 'el1', 1813.0, 686.0, 'exp8_d7_chipE_ch2',
     'Very clean negative-polarity signal, CV ~5 %.'),
    ('Exp7/ChipE/chipE_ch3_baseline.csv', 'el2', 1805.0, 606.0, 'exp7_chipE_ch3',
     'Low noise floor; ~10 % of raw detections were noise before the gate.'),
    ('Exp7/ChipC_D/chipC_ch1_baseline.csv', 'el2', 1393.0, 437.0, 'exp7_chipC_ch1',
     'Perfectly regular, every beat detected on both electrodes at RR ~1550 ms. '
     'The published 1393 ms is ~12 % lower: reference-window discrepancy, '
     'not a detection error. Wider RR tolerance.'),
    ('Exp5/Day7/chipA/chipA_ch1_baseline.csv', 'el2', 799.0, 413.0, 'exp5_chipA_ch1',
     'Fast rhythm (~75 BPM), negative polarity, 220 beats in 180 s.'),
    ('Exp5/Day7/ChipB/chipB_ch3_baseline.csv', 'el2', 2333.0, 711.0, 'exp5_chipB_ch3',
     'Slow regular rhythm.'),
    # Low-SNR biphasic case (lab recording, Apr 2026, not from the paper).
    # el1 spikes ~15 µV on ~10 µV noise: every real beat sits at SNR 1.1–2.3.
    # Reference is the same recording's el2 (clean, 220 beats, RR 811 ms,
    # CV 6 %), measured by this pipeline — a cross-electrode reference, not
    # a published one. FPD reference = el2's 492 ms is NOT used (different
    # electrode, different T-wave); FPD tolerance is skipped for this file.
    ('__LOCAL__studio_MR/Baseline/chipA_ch1_baseline.csv', 'el1', 811.0, None, 'lab_lowsnr_chipA_ch1',
     'Low-SNR biphasic el1; reference RR from el2 of the same recording. '
     'The first noise-floor gate (fixed floor 1.5) removed 44/208 real beats here.'),
    ('Exp6/CHIPC/chipC_ch1_baseline.csv', 'el1', 1835.0, 555.0, 'exp6_chipC_ch1',
     'Mixed polarity. On 60-90 s windows the ungated detector doubles the '
     'count (RR ~830 ms); the gate restores ~1740 ms.'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel, ch, rr, fpd, label, notes in CORPUS:
        src = (ROOT / rel[len('__LOCAL__'):]) if rel.startswith('__LOCAL__') else DATASET / rel
        if not src.exists():
            print(f"  !! missing {src}", file=sys.stderr)
            continue
        meta, df = load_csv(str(src))
        fs = float(meta['sample_rate'])
        t0, dur = WINDOW_OVERRIDE.get(label, (T0_S, DUR_S))
        i0 = int(t0 * fs)
        i1 = min(len(df), i0 + int(dur * fs))
        sig = df[ch].values[i0:i1].astype(np.float32)
        arrays = {'signal': sig, 'fs': fs}
        if rel.startswith('__LOCAL__'):
            # Lab recordings: also store the other electrode so tests can
            # compare beat POSITIONS (missed / spurious), not just RR.
            other = 'el2' if ch == 'el1' else 'el1'
            arrays['reference_signal'] = df[other].values[i0:i1].astype(np.float32)
        np.savez_compressed(OUT / f'{label}.npz', **arrays)
        entry = {
            'label': label, 'file': f'{label}.npz', 'source': rel.replace('__LOCAL__', ''),
            'channel': ch, 'fs': fs, 't0_s': t0, 'duration_s': len(sig) / fs,
            'reference': {'rr_ms': rr, 'fpd_ms': fpd,
                          'source': ('same recording, electrode el2 (pipeline measurement)'
                                     if rel.startswith('__LOCAL__') else
                                     'Visone et al. 2023 Excel (data_reference/ground_truth.json)')},
            'notes': notes,
        }
        manifest.append(entry)
        print(f"  {label:20s} {len(sig):7d} samples @ {fs:.0f} Hz  "
              f"{(OUT / f'{label}.npz').stat().st_size / 1024:6.0f} KB")
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"wrote {len(manifest)} fixtures + manifest.json to {OUT}")


if __name__ == '__main__':
    main()
