"""Review model for the operators (v3.17, Oct 2026) — no Qt, fully testable.

One ``ReviewState`` per recording and input: the depolarisations and
repolarisations proposed by the software, the detections the vetoes removed
(restorable), the noise-burst windows, and the operator's corrections. Every
correction is an operation with undo / redo; the measures are recomputed
from the current state with the conventions of the lab's Results workbooks
(sheets ``CHIP X`` and ``CHIP X raw``):

* beating period = mean of the intervals between consecutive depolarisations,
  SD with n − 1, CV = SD / mean;
* FPD = mean of the per-beat FPD (repolarisation − depolarisation);
* FPDc (Fridericia) = mean FPD / mean BP^(1/3), on the means;
* depolarisation amplitude = peak-to-peak of the *unfiltered* trace within
  ±10 ms of the spike, in µV (on the lab's chip A ch1 this reproduces the
  workbook: 368 vs 367 µV; the filtered trace gives 281);
* ``%`` and ``variation to baseline`` against a reference measures dict.

Repolarisation propagation (the operator's request): when the operator moves
one repolarisation marker, ``set_repol(..., propagate=True)`` infers the
criterion — polarity of the chosen wave and its latency from the
depolarisation — and applies it to every other beat of the recording that has
a wave of that polarity near that latency; beats without one keep their
marker and are flagged ``ambiguo``.

Corrections are saved per file and per input in a JSON sidecar
(``<recording>.review.json``), in seconds, so they survive re-analysis.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sg

logger = logging.getLogger(__name__)

REVIEW_SUFFIX = '.review.json'
SCHEMA = '1'
MATCH_TOL_S = 0.050           # a click within this of an existing marker refers to it
STATUS_TO_REVIEW, STATUS_APPROVED, STATUS_CLOSED = 'da_rivedere', 'approvato', 'chiuso'
STATUSES = (STATUS_TO_REVIEW, STATUS_APPROVED, STATUS_CLOSED)


def operator_name():
    """Who approves: ``CARDIAC_FP_OPERATOR`` when set, else the login name."""
    import getpass
    import os
    return os.environ.get('CARDIAC_FP_OPERATOR') or getpass.getuser()


class RecordingClosedError(PermissionError):
    """The recording was closed by the operator: no correction is accepted until it is reopened."""
MIN_RR_S = 0.100              # an added beat may not sit closer than this to another


@dataclass
class BeatMark:
    dep: int                                # sample index of the depolarisation
    repol: int | None = None                # sample index of the repolarisation peak (None = not measurable)
    source: str = 'auto'                    # 'auto' | 'corretto' | 'aggiunto' | 'ripristinato'
    repol_source: str = 'auto'              # 'auto' | 'corretto' | 'propagato' | 'assente' | 'ambiguo'
    note: str = ''


@dataclass
class _Snapshot:
    beats: list
    vetoed: list
    polarity: int | None
    criterion: dict | None
    removed: list


class ReviewState:
    """Beats, repolarisations and corrections of one recording / input."""

    def __init__(self, fs, filtered, raw=None, *, channel='', bursts=(), vetoed=(), gain=1.0, source_file=''):
        self.fs = float(fs)
        self.filtered = np.asarray(filtered, dtype=float)
        self.raw = np.asarray(raw, dtype=float) if raw is not None else None
        self.channel = str(channel)
        self.bursts = [(int(a), int(b)) for a, b in bursts]
        self.vetoed = sorted(int(v) for v in vetoed)
        self.gain = float(gain)
        self.source_file = str(source_file)
        self.beats: list[BeatMark] = []
        self.removed: list[int] = []                # software detections the operator removed (sample indices)
        # review workflow: to review → approved (tick) → closed (locked); saved in the sidecar
        self.status = STATUS_TO_REVIEW
        self.approved_by = ''
        self.approved_at = ''
        self.closed_at = ''
        self.polarity: int | None = None            # repolarisation polarity chosen by the operator (+1 / -1)
        self.criterion: dict | None = None          # last propagated criterion
        self.notes = ''
        self._history: list[_Snapshot] = []
        self._future: list[_Snapshot] = []

    # ── construction ───────────────────────────────────────────────────────

    @classmethod
    def from_result(cls, result, config=None):
        """Software proposal from an ``analyze_single_file`` result (beats, per-beat repolarisation,
        detections removed by the vetoes, burst windows)."""
        from .config import AnalysisConfig
        cfg = config or AnalysisConfig()
        md = result.get('metadata', {}) or {}
        fs = float(md.get('sample_rate'))
        x = np.asarray(result['filtered_signal'], dtype=float)
        det = result.get('detection_info') or {}
        vet = det.get('post_detection_vetoes') or {}
        removed = []
        for key in ('weak_population_split', 'burst_veto', 'min_rr'):
            removed += list((vet.get(key) or {}).get('removed') or [])
        windows = (vet.get('burst_veto') or {}).get('windows')
        if windows is None:
            from .detection_vetoes import burst_windows
            windows = burst_windows(x, fs, cfg.beat_detection)
        st = cls(fs, x, result.get('raw_signal'), channel=(result.get('file_info') or {}).get('analyzed_channel', ''),
                 bursts=windows, vetoed=removed, gain=float(getattr(cfg, 'amplifier_gain', 1.0) or 1.0),
                 source_file=md.get('filepath', ''))
        deps = [int(i) for i in np.asarray(result.get('beat_indices_raw', []), dtype=int)]
        repol = {}
        bf = np.asarray(result.get('beat_indices_fpd', []), dtype=int)
        for k, p in enumerate(result.get('all_params') or []):
            g = p.get('repol_peak_global_idx')
            if k < len(bf) and g is not None and np.isfinite(g):
                repol[int(bf[k])] = int(g)
        st.beats = [BeatMark(d, repol.get(d), 'auto', 'auto' if d in repol else 'assente') for d in deps]
        st.beats.sort(key=lambda b: b.dep)
        return st

    # ── undo / redo ────────────────────────────────────────────────────────

    def _snapshot(self):
        return _Snapshot(copy.deepcopy(self.beats), list(self.vetoed), self.polarity, copy.deepcopy(self.criterion),
                         list(self.removed))

    def _restore(self, s):
        self.beats, self.vetoed, self.polarity, self.criterion = copy.deepcopy(s.beats), list(s.vetoed), s.polarity, copy.deepcopy(s.criterion)
        self.removed = list(s.removed)

    def _begin(self):
        if self.locked:
            raise RecordingClosedError('registrazione chiusa: sbloccarla per correggere')
        self._history.append(self._snapshot())
        self._future.clear()

    # ── approval / closure ─────────────────────────────────────────────────

    @property
    def locked(self):
        return self.status == STATUS_CLOSED

    @property
    def approved(self):
        return self.status in (STATUS_APPROVED, STATUS_CLOSED)

    def approve(self, operator=None):
        """Tick the recording as reviewed and approved (no effect when closed)."""
        from datetime import datetime
        if self.locked:
            return
        self.status = STATUS_APPROVED
        self.approved_by = operator or operator_name()
        self.approved_at = datetime.now().astimezone().isoformat(timespec='seconds')

    def unapprove(self):
        if self.locked:
            return
        self.status, self.approved_by, self.approved_at = STATUS_TO_REVIEW, '', ''

    def close(self, operator=None):
        """Close the recording: approved (if not yet) and locked against corrections."""
        from datetime import datetime
        if not self.approved:
            self.approve(operator)
        self.status = STATUS_CLOSED
        self.closed_at = datetime.now().astimezone().isoformat(timespec='seconds')

    def reopen(self):
        """Unlock a closed recording (it stays approved)."""
        if self.locked:
            self.status, self.closed_at = STATUS_APPROVED, ''

    def status_text(self):
        if self.status == STATUS_CLOSED:
            return f"chiuso il {self.closed_at[:16].replace('T', ' ')}" + (f" ({self.approved_by})" if self.approved_by else '')
        if self.status == STATUS_APPROVED:
            return f"approvato da {self.approved_by or '?'} il {self.approved_at[:16].replace('T', ' ')}"
        return 'da rivedere'

    @property
    def can_undo(self):
        return bool(self._history)

    @property
    def can_redo(self):
        return bool(self._future)

    def undo(self):
        if not self._history:
            return False
        self._future.append(self._snapshot())
        self._restore(self._history.pop())
        return True

    def redo(self):
        if not self._future:
            return False
        self._history.append(self._snapshot())
        self._restore(self._future.pop())
        return True

    # ── lookup ─────────────────────────────────────────────────────────────

    def n_beats(self):
        return len(self.beats)

    def deps(self):
        return np.array([b.dep for b in self.beats], dtype=int)

    def beat_near(self, sample, tol_s=MATCH_TOL_S):
        """Index of the beat whose depolarisation is within ``tol_s`` of ``sample``, or None."""
        if not self.beats:
            return None
        d = self.deps()
        k = int(np.argmin(np.abs(d - sample)))
        return k if abs(d[k] - sample) <= tol_s * self.fs else None

    def repol_near(self, sample, tol_s=MATCH_TOL_S):
        best = None
        for k, b in enumerate(self.beats):
            if b.repol is not None and abs(b.repol - sample) <= tol_s * self.fs:
                if best is None or abs(b.repol - sample) < abs(self.beats[best].repol - sample):
                    best = k
        return best

    def vetoed_near(self, sample, tol_s=MATCH_TOL_S):
        for v in self.vetoed:
            if abs(v - sample) <= tol_s * self.fs:
                return v
        return None

    # ── operations (each undoable) ─────────────────────────────────────────

    def _snap_to_spike(self, sample, half_s=0.010):
        h = int(half_s * self.fs)
        a, b = max(0, sample - h), min(len(self.filtered), sample + h + 1)
        return int(a + np.argmax(np.abs(self.filtered[a:b])))

    def add_beat(self, sample, *, snap=True, source='aggiunto'):
        """Add a depolarisation at ``sample`` (snapped to the largest |x| within ±10 ms). Returns its index,
        or None when a beat already sits within MIN_RR_S."""
        sample = int(sample)
        if snap:
            sample = self._snap_to_spike(sample)
        if any(abs(b.dep - sample) < MIN_RR_S * self.fs for b in self.beats):
            return None
        self._begin()
        self.beats.append(BeatMark(sample, None, source, 'assente'))
        self.beats.sort(key=lambda b: b.dep)
        k = next(i for i, b in enumerate(self.beats) if b.dep == sample)
        self.propose_repol(k, record=False)
        return k

    def remove_beat(self, k):
        self._begin()
        if self.beats[k].source == 'auto':
            self.removed.append(self.beats[k].dep)
        del self.beats[k]

    def changes(self):
        """What the operator changed with respect to the software proposal: counts of removed, added,
        moved and restored depolarisations, corrected and propagated repolarisations, the chosen polarity,
        and ``corrected`` (True when anything differs)."""
        c = {'tolti': len(self.removed),
             'aggiunti': sum(1 for b in self.beats if b.source == 'aggiunto'),
             'spostati': sum(1 for b in self.beats if b.source == 'corretto'),
             'ripristinati': sum(1 for b in self.beats if b.source == 'ripristinato'),
             'ripol_corrette': sum(1 for b in self.beats if b.repol_source == 'corretto'),
             'ripol_propagate': sum(1 for b in self.beats if b.repol_source == 'propagato'),
             'polarita': self.polarity}
        c['corrected'] = any(v for k, v in c.items() if k != 'polarita') or self.polarity is not None
        return c

    def move_beat(self, k, sample, *, snap=True):
        sample = int(sample)
        if snap:
            sample = self._snap_to_spike(sample)
        self._begin()
        b = self.beats[k]
        b.dep, b.source = sample, 'corretto'
        if b.repol is not None and b.repol <= sample:
            b.repol, b.repol_source = None, 'assente'
        self.beats.sort(key=lambda bb: bb.dep)

    def restore_vetoed(self, sample):
        """Bring back a detection the vetoes removed (the one within MATCH_TOL_S of ``sample``)."""
        v = self.vetoed_near(sample)
        if v is None:
            return None
        self._begin()
        self.vetoed.remove(v)
        self.beats.append(BeatMark(v, None, 'ripristinato', 'assente'))
        self.beats.sort(key=lambda b: b.dep)
        k = next(i for i, b in enumerate(self.beats) if b.dep == v)
        self.propose_repol(k, record=False)
        return k

    def set_repol(self, k, sample, *, snap=True, propagate=False, tol_frac=0.15, tol_min_s=0.100):
        """Place the repolarisation of beat ``k`` at ``sample`` (snapped to the nearest local extremum).
        With ``propagate`` the same criterion (polarity, latency) is applied to every other beat; returns the
        propagation report (dict) or None."""
        b = self.beats[k]
        sample = int(sample)
        if snap:
            sample = self._snap_to_extremum(sample)
        if sample <= b.dep:
            raise ValueError('la ripolarizzazione deve seguire la depolarizzazione')
        self._begin()
        b.repol, b.repol_source = sample, 'corretto'
        if not propagate:
            return None
        return self._propagate(k, tol_frac, tol_min_s, record=False)

    def clear_repol(self, k):
        self._begin()
        self.beats[k].repol, self.beats[k].repol_source = None, 'assente'

    def set_polarity(self, sign, *, propagate=True, tol_frac=0.15, tol_min_s=0.100):
        """Repolarisation polarity (+1 positive peak, -1 negative) for the whole recording; with ``propagate``
        every beat is re-pointed to the most prominent extremum of that sign near the current median latency."""
        self._begin()
        self.polarity = int(np.sign(sign)) or None
        if not propagate or self.polarity is None:
            return None
        # first pass: the most prominent wave of that sign anywhere in each beat's window
        rep = self._apply_criterion({'polarita': self.polarity}, skip=None)
        lat = self._median_latency()
        if lat is not None:
            proms = [self._prominence_at(b.repol, self.polarity) for b in self.beats if b.repol is not None]
            self.criterion = {'polarita': self.polarity, 'latenza_s': lat, 'tolleranza_s': max(tol_min_s, tol_frac * lat),
                              'prominenza_minima': 0.3 * float(np.median(proms)) if proms else None}
            # second pass: the same wave, now near the common latency (drops stray picks)
            rep2 = self._apply_criterion(self.criterion, skip=None)
            rep['spostati'] = sum(1 for b in self.beats if b.repol_source == 'propagato')
            rep['ambigui'] = rep2['ambigui']
            rep['invariati'] = len(self.beats) - rep['spostati'] - rep['ambigui']
            rep['criterio'] = dict(self.criterion)
        return rep

    def propagate_from(self, k, tol_frac=0.15, tol_min_s=0.100):
        """Apply beat ``k``'s repolarisation choice (polarity, latency) to all the other beats."""
        self._begin()
        return self._propagate(k, tol_frac, tol_min_s, record=False)

    # ── repolarisation search ──────────────────────────────────────────────

    def _snap_to_extremum(self, sample, half_s=0.030):
        x = self.filtered
        h = int(half_s * self.fs)
        a, b = max(1, sample - h), min(len(x) - 1, sample + h + 1)
        seg = x[a:b]
        d = np.diff(np.sign(np.diff(seg)))
        ext = np.where(d != 0)[0] + 1
        if len(ext) == 0:
            return int(sample)
        return int(a + ext[np.argmin(np.abs(ext + a - sample))])

    def _median_latency(self):
        lat = [(b.repol - b.dep) / self.fs for b in self.beats if b.repol is not None]
        return float(np.median(lat)) if lat else None

    def _noise(self):
        """Robust noise of the filtered trace (1.4826 × MAD), cached."""
        if getattr(self, '_noise_cache', None) is None:
            x = self.filtered
            self._noise_cache = float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12)
        return self._noise_cache

    def _extrema(self, lo, hi, sign, min_prom=None):
        """Local extrema of the given sign in [lo, hi) with their prominence (filtered trace). Only extrema
        standing out of the noise: prominence ≥ max(3 × noise, ``min_prom``)."""
        x = self.filtered
        lo, hi = max(0, int(lo)), min(len(x), int(hi))
        if hi - lo < 5:
            return np.array([], dtype=int), np.array([])
        seg = sign * x[lo:hi]
        floor = max(3.0 * self._noise(), float(min_prom or 0.0))
        pk, pr = sg.find_peaks(seg, prominence=floor, distance=max(1, int(0.03 * self.fs)))
        return pk + lo, pr['prominences']

    def _prominence_at(self, sample, sign, half_s=0.3):
        """Prominence of the extremum of ``sign`` at ``sample`` (0 if there is none)."""
        h = int(half_s * self.fs)
        pk, pr = self._extrema(sample - h, sample + h, sign, min_prom=0.0)
        if len(pk) == 0:
            return 0.0
        j = int(np.argmin(np.abs(pk - sample)))
        return float(pr[j]) if abs(pk[j] - sample) <= int(0.02 * self.fs) else 0.0

    def _window(self, k):
        """Search window of beat k: from 100 ms after the spike to 50 ms before the next beat (at most 85 % RR)."""
        b = self.beats[k]
        nxt = self.beats[k + 1].dep if k + 1 < len(self.beats) else len(self.filtered)
        rr = nxt - b.dep
        return b.dep + int(0.1 * self.fs), min(nxt - int(0.05 * self.fs), b.dep + int(0.85 * rr))

    def _search(self, k, criterion):
        """Best extremum for beat ``k`` under ``criterion`` (polarity; latency ± tolerance when given; minimum
        prominence as a fraction of the reference wave) or None."""
        b = self.beats[k]
        sign = criterion['polarita']
        lo, hi = self._window(k)
        lat, tol = criterion.get('latenza_s'), criterion.get('tolleranza_s')
        if lat is not None and tol is not None:
            lo = max(lo, b.dep + int((lat - tol) * self.fs))
            hi = min(hi, b.dep + int((lat + tol) * self.fs))
        m = int(0.05 * self.fs)                      # margin so that a wave at the window edge is still a peak
        pk, pr = self._extrema(lo - m, hi + m, sign, min_prom=criterion.get('prominenza_minima'))
        keep = (pk >= lo) & (pk < hi)
        pk, pr = pk[keep], pr[keep]
        if len(pk) == 0:
            return None
        return int(pk[int(np.argmax(pr))])

    def _propagate(self, k, tol_frac, tol_min_s, record=True):
        b = self.beats[k]
        if b.repol is None:
            return None
        if record:
            self._begin()
        lat = (b.repol - b.dep) / self.fs
        sign = int(np.sign(self.filtered[b.repol])) or 1
        prom = self._prominence_at(b.repol, sign)
        if prom < 3.0 * self._noise():
            # no recognisable wave at the chosen point: the criterion cannot be inferred, nothing is moved
            return {'spostati': 0, 'ambigui': len(self.beats) - 1, 'invariati': 0, 'criterio': None,
                    'motivo': 'nessuna onda riconoscibile nel punto scelto: criterio non deducibile'}
        self.criterion = {'polarita': sign, 'latenza_s': lat, 'tolleranza_s': max(tol_min_s, tol_frac * lat),
                          'prominenza_minima': 0.3 * prom, 'battito_di_riferimento': k}
        self.polarity = sign
        return self._apply_criterion(self.criterion, skip=k)

    def _apply_criterion(self, criterion, skip):
        rep = {'spostati': 0, 'ambigui': 0, 'invariati': 0, 'criterio': dict(criterion)}
        for j, bj in enumerate(self.beats):
            if j == skip:
                continue
            found = self._search(j, criterion)
            if found is None:
                bj.repol_source = 'ambiguo' if bj.repol is not None else 'assente'
                rep['ambigui'] += 1
            elif bj.repol == found:
                rep['invariati'] += 1
            else:
                bj.repol, bj.repol_source = found, 'propagato'
                rep['spostati'] += 1
        return rep

    def propose_repol(self, k, record=True):
        """Repolarisation for a beat without one, from the recording's current criterion (operator's polarity
        and median latency) or, failing that, the most prominent extremum in 150 ms – 85 % of the RR."""
        if record:
            self._begin()
        b = self.beats[k]
        crit = self.criterion
        if crit is None:
            lat = self._median_latency()
            sign = self.polarity
            if sign is None:
                signs = [int(np.sign(self.filtered[bb.repol])) for bb in self.beats if bb.repol is not None]
                sign = int(np.sign(sum(signs))) or 1 if signs else 1
            if lat is not None:
                crit = {'polarita': sign, 'latenza_s': lat, 'tolleranza_s': max(0.1, 0.15 * lat)}
        found = self._search(k, crit) if crit else None
        if found is None:
            lo, hi = self._window(k)
            best = None
            for s in (crit['polarita'],) if crit else (1, -1):
                pk, pr = self._extrema(lo, hi, s)
                if len(pk) and (best is None or pr.max() > best[1]):
                    best = (int(pk[int(np.argmax(pr))]), float(pr.max()))
            found = best[0] if best else None
        b.repol = found
        b.repol_source = 'propagato' if found is not None else 'assente'
        return found

    # ── measures ───────────────────────────────────────────────────────────

    def _amp_uV(self, dep, half_s=0.010):
        src = self.raw if self.raw is not None else self.filtered
        h = int(half_s * self.fs)
        w = src[max(0, dep - h):min(len(src), dep + h + 1)]
        return float((w.max() - w.min()) * 1e6), float(w.max() * self.gain), float(w.min() * self.gain)

    def per_beat_rows(self):
        """One row per beat like the sheet ``CHIP X raw``: n, dep_s, bp_s, rep_s, fpd_s, v_up, v_down, amp_uV,
        stato (depolarisation source), stato_rip (repolarisation source)."""
        rows = []
        d = self.deps()
        for k, b in enumerate(self.beats):
            amp, up, dn = self._amp_uV(b.dep)
            rows.append({'n': k + 1, 'dep_s': b.dep / self.fs,
                         'bp_s': (d[k + 1] - b.dep) / self.fs if k + 1 < len(d) else None,
                         'rep_s': b.repol / self.fs if b.repol is not None else None,
                         'fpd_s': (b.repol - b.dep) / self.fs if b.repol is not None else None,
                         'v_up': up, 'v_down': dn, 'amp_uV': amp, 'stato': b.source, 'stato_rip': b.repol_source})
        return rows

    def drift(self):
        """Drift of the rate and the FPD over the recording from the current beats (``drift.rate_drift``)."""
        from .drift import rate_drift
        rows = self.per_beat_rows()
        return rate_drift([r['dep_s'] for r in rows], [r['fpd_s'] if r['fpd_s'] is not None else np.nan for r in rows])

    def measures(self, reference=None, exclude_gaps=True, gap_factor=1.5):
        """The columns of the sheet ``CHIP X`` (seconds, %, µV). With ``exclude_gaps`` the intervals longer than
        ``gap_factor`` × the median (a missing beat) are left out of the beating period; the count is reported.
        ``reference``: the measures of the baseline → ``*_pct`` and ``*_var`` (variation to baseline)."""
        rows = self.per_beat_rows()
        bp = np.array([r['bp_s'] for r in rows if r['bp_s'] is not None], dtype=float)
        n_gap = 0
        if exclude_gaps and len(bp) >= 3:
            med = np.median(bp)
            keep = bp < gap_factor * med
            n_gap = int((~keep).sum())
            bp = bp[keep]
        fpd = np.array([r['fpd_s'] for r in rows if r['fpd_s'] is not None], dtype=float)
        amp = np.array([r['amp_uV'] for r in rows], dtype=float)
        m = {'n_battiti': len(rows), 'n_intervalli': int(len(bp)), 'intervalli_esclusi': n_gap, 'n_fpd': int(len(fpd)),
             'bp_s': float(bp.mean()) if len(bp) else None,
             'bp_sd_s': float(bp.std(ddof=1)) if len(bp) > 1 else None,
             'bp_cv_pct': float(bp.std(ddof=1) / bp.mean() * 100) if len(bp) > 1 and bp.mean() > 0 else None,
             'fpd_s': float(fpd.mean()) if len(fpd) else None,
             'amp_uV': float(amp.mean()) if len(amp) else None,
             'n_vetoed': len(self.vetoed), 'burst_frac': self.burst_fraction()}
        m['fpdc_s'] = m['fpd_s'] / m['bp_s'] ** (1 / 3) if m['fpd_s'] is not None and m['bp_s'] else None
        d = self.drift() if len(rows) >= 3 else None
        m['drift'] = d
        m['drift_pct'] = d['variazione_pct'] if d and d.get('pendenza_bpm_min') is not None else None
        m['drift_bpm_min'] = d['pendenza_bpm_min'] if d else None
        m['drift_ok'] = bool(d and d.get('affidabile'))
        m['fpd_drift_pct'] = (d['fpd']['variazione_pct'] if d and d.get('fpd') and d['fpd'].get('pendenza_ms_min') is not None
                              else None)
        if reference:
            for key in ('bp_s', 'fpd_s', 'fpdc_s', 'amp_uV'):
                base = reference.get(key)
                if m.get(key) is not None and base:
                    m[key + '_pct'] = m[key] / base * 100
                    m[key + '_var'] = m[key + '_pct'] - 100
        return m

    def problems(self, gap_factor=1.5, close_factor=0.5):
        """Stretches of the trace the operator should look at, as dicts ``{'tipo', 'start', 'end', 'k'}``
        (sample indices, ``k`` the beat before the stretch): ``battito_mancante`` — an interval longer than
        ``gap_factor`` × the median interval (a depolarisation probably missing between two beats; the same rule
        excludes it from the beating period); ``battiti_vicini`` — an interval shorter than ``close_factor`` ×
        the median (a detection in excess); ``ripolarizzazione_assente`` — a beat without a repolarisation, from
        its depolarisation to the next one (or one median interval). Needs at least 3 intervals for the
        interval rules."""
        d = self.deps()
        out = []
        n = len(self.filtered)
        rr = np.diff(d) if len(d) > 1 else np.array([], dtype=float)
        med = float(np.median(rr)) if len(rr) >= 3 else None
        if med:
            for k, r in enumerate(rr):
                if r > gap_factor * med:
                    out.append({'tipo': 'battito_mancante', 'start': int(d[k]), 'end': int(d[k + 1]), 'k': k})
                elif r < close_factor * med:
                    out.append({'tipo': 'battiti_vicini', 'start': int(d[k]), 'end': int(d[k + 1]), 'k': k})
        span = int(med) if med else int(1.0 * self.fs)
        for k, b in enumerate(self.beats):
            if b.repol is None:
                end = int(d[k + 1]) if k + 1 < len(d) else min(n - 1, b.dep + span)
                out.append({'tipo': 'ripolarizzazione_assente', 'start': int(b.dep), 'end': int(end), 'k': k})
        out.sort(key=lambda x: (x['start'], x['tipo']))
        return out

    def burst_fraction(self):
        n = len(self.filtered)
        return float(sum(e - s for s, e in self.bursts) / n) if n else 0.0

    # ── persistence ────────────────────────────────────────────────────────

    def to_dict(self):
        crit = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v)
                for k, v in (self.criterion or {}).items()} or None
        return {'canale': self.channel, 'fs': self.fs, 'polarita': self.polarity, 'criterio': crit,
                'note': self.notes, 'stato': self.status, 'approvato_da': self.approved_by,
                'approvato_il': self.approved_at, 'chiuso_il': self.closed_at,
                'battiti': [{'dep_s': b.dep / self.fs, 'rep_s': b.repol / self.fs if b.repol is not None else None,
                             'stato': b.source, 'stato_rip': b.repol_source, 'nota': b.note} for b in self.beats],
                'tolti_dai_veti_s': [v / self.fs for v in self.vetoed],
                'tolti_s': [v / self.fs for v in self.removed],
                'burst_s': [[s / self.fs, e / self.fs] for s, e in self.bursts]}

    def apply_dict(self, d):
        """Load saved corrections (times in seconds) into this state."""
        fs = self.fs
        self.beats = [BeatMark(int(round(b['dep_s'] * fs)), int(round(b['rep_s'] * fs)) if b.get('rep_s') is not None else None,
                               b.get('stato', 'auto'), b.get('stato_rip', 'auto'), b.get('nota', '')) for b in d.get('battiti', [])]
        self.beats.sort(key=lambda b: b.dep)
        self.vetoed = sorted(int(round(v * fs)) for v in d.get('tolti_dai_veti_s', []))
        self.removed = [int(round(v * fs)) for v in d.get('tolti_s', [])]
        self.polarity = d.get('polarita')
        self.criterion = d.get('criterio')
        self.notes = d.get('note', '')
        self.status = d.get('stato') if d.get('stato') in STATUSES else STATUS_TO_REVIEW
        self.approved_by = d.get('approvato_da', '') or ''
        self.approved_at = d.get('approvato_il', '') or ''
        self.closed_at = d.get('chiuso_il', '') or ''
        self._history.clear()
        self._future.clear()

    @staticmethod
    def sidecar_path(recording_path):
        p = Path(recording_path)
        return p.with_name(p.name + REVIEW_SUFFIX)

    def save(self, recording_path=None):
        """Write this input's corrections into the recording's sidecar (other inputs are kept)."""
        path = self.sidecar_path(recording_path or self.source_file)
        data = {'schema': SCHEMA, 'ingressi': {}}
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(old, dict) and old.get('schema') == SCHEMA:
                    data = old
            except (OSError, ValueError) as exc:
                logger.warning('review sidecar %s unreadable, overwriting: %s', path, exc)
        data['ingressi'][self.channel or 'el1'] = self.to_dict()
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
        return path

    @classmethod
    def _read_sidecar(cls, recording_path):
        path = cls.sidecar_path(recording_path)
        if not path.exists():
            return None, path
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None, path
        return (data if isinstance(data, dict) and data.get('schema') == SCHEMA else None), path

    @classmethod
    def chosen_electrode(cls, recording_path):
        """The input the operator chose for this recording's measures ('el1', 'el2', an electrode label),
        saved by ``save_choice``; None when not chosen."""
        data, _ = cls._read_sidecar(recording_path)
        return (data or {}).get('elettrodo_scelto') or None

    @classmethod
    def saved_status(cls, recording_path):
        """{input label: status} of the inputs saved in the sidecar (without loading the data)."""
        data, _ = cls._read_sidecar(recording_path)
        out = {}
        for label, entry in ((data or {}).get('ingressi') or {}).items():
            st = (entry or {}).get('stato')
            out[label] = st if st in STATUSES else STATUS_TO_REVIEW
        return out

    @classmethod
    def save_choice(cls, recording_path, electrode):
        """Record in the sidecar which input is used for the measures (``None`` removes the choice)."""
        data, path = cls._read_sidecar(recording_path)
        data = data or {'schema': SCHEMA, 'ingressi': {}}
        if electrode:
            data['elettrodo_scelto'] = str(electrode)
        else:
            data.pop('elettrodo_scelto', None)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
        tmp.replace(path)
        return path

    @classmethod
    def load_into(cls, state, recording_path=None):
        """Apply the saved corrections of ``state.channel`` (if any) to ``state``; returns True when found."""
        path = cls.sidecar_path(recording_path or state.source_file)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding='utf-8'))
        entry = (data.get('ingressi') or {}).get(state.channel or 'el1')
        if not entry:
            return False
        state.apply_dict(entry)
        return True

    # ── interplay with the pipeline ────────────────────────────────────────

    def recompute(self, result, config=None):
        """Re-run the post-detection pipeline on the current beats (``analyze.recompute_from_beats``), keep the
        operator's repolarisations, refresh the automatic ones. Returns the new result."""
        from .analyze import recompute_from_beats
        new = recompute_from_beats(result, self.deps(), config=config)
        bf = np.asarray(new.get('beat_indices_fpd', []), dtype=int)
        auto = {}
        for k, p in enumerate(new.get('all_params') or []):
            g = p.get('repol_peak_global_idx')
            if k < len(bf) and g is not None and np.isfinite(g):
                auto[int(bf[k])] = int(g)
        for b in self.beats:
            if b.repol_source in ('auto', 'assente'):
                b.repol = auto.get(b.dep)
                b.repol_source = 'auto' if b.repol is not None else 'assente'
        return new


def state_to_json(state):
    return json.dumps(state.to_dict(), ensure_ascii=False)


__all__ = ['BeatMark', 'ReviewState', 'REVIEW_SUFFIX', 'state_to_json']
