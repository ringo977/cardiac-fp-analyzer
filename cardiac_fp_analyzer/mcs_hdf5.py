"""
mcs_hdf5.py — Reader for Multi Channel Systems HDF5 raw-data files.

The MCS "RawData" protocol (HDF5 MCS Raw Data Definition, protocol
version 3, April 2019) stores one experiment per file:

    /Data                                   MeaLayout, MeaName, ProgramName, Date, DateInTicks
      /Recording_<r>                        Duration (µs), TimeStamp
        /AnalogStream/Stream_<s>            Label, DataSubType ('Electrode', 'Auxiliary' …)
           ChannelData                      int32 matrix, one row per channel, one column per sample
           ChannelDataTimeStamps            k × 3: (first time stamp µs, first column, last column) per segment
           InfoChannel                      per channel: Label, Unit, Exponent, ADZero, Tick (µs),
                                            ConversionFactor, RowIndex, filters
        /EventStream/Stream_<s>             events: EventEntity_<e> (time µs, duration µs, …) + InfoEvent
        /SegmentStream/Stream_<s>           cut-outs around events (spike wave forms, averages)
        /TimeStampStream/Stream_<s>         time stamps only (spike detector) + InfoTimeStamp

Physical value of a sample:
    y = (ChannelData[RowIndex, t] - ADZero) * ConversionFactor * 10**Exponent   [Unit]
Sample time:  t_index * Tick  (µs).

Only the analog streams are needed by the pipeline. Events are read so a
paced recording (stimulator running, stimulus times in an EventStream) can
be recognised; spike time stamps are read as an external beat list.

The electrode signal is read in blocks along time (the file is chunked that
way) and, when sampled above ``MAX_SAMPLE_RATE``, decimated on the fly to
about ``TARGET_SAMPLE_RATE`` with a linear-phase FIR (Kaiser window, cut-off
0.45 of the output rate), with no delay: output sample m is centred on input
sample m·q. A 64-channel, 6-minute, 20 kHz recording (1.6 GB of int32 in the
file) therefore needs about 20 MB per block plus the decimated output
(64 × 2 kHz × float32 ≈ 0.5 MB per second).

h5py is an optional dependency (``pip install cardiac-fp-analyzer[mcs]``).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_SAMPLE_RATE = 2000.0
MAX_SAMPLE_RATE = 3000.0
SUFFIXES = ('.h5', '.hdf5', '.hdf')
# Event sub-types written by the stimulator or the digital input port: a
# recording with one of these streams is treated as paced.
STIMULUS_SUBTYPES = ('StgSideband', 'DigitalPort', 'StimulatorSideband')
_BLOCK = 1 << 16          # samples per block read from ChannelData


def _h5py():
    try:
        import h5py
    except ImportError as e:   # pragma: no cover - depends on the environment
        raise ImportError("Reading MCS HDF5 files needs h5py: pip install h5py "
                          "(or cardiac-fp-analyzer[mcs])") from e
    return h5py


def _s(v):
    """Attribute / compound field to a clean str."""
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode('ascii', errors='replace')
    elif isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else v.tolist()
        return _s(v) if not isinstance(v, list) else str(v)
    return str(v).strip()


def is_mcs_hdf5(filepath):
    """True when the file is an HDF5 file carrying the MCS RawData protocol."""
    p = Path(filepath)
    if p.suffix.lower() not in SUFFIXES or not p.is_file():
        return False
    try:
        h5py = _h5py()
        with h5py.File(p, 'r') as h:
            return _s(h.attrs.get('McsHdf5ProtocolType', '')) == 'RawData' and 'Data' in h
    except (OSError, ImportError):
        return False


def _info_rows(ds):
    """Rows of an Info* compound dataset as dicts of clean values."""
    out = []
    for row in ds[:]:
        d = {}
        for name in ds.dtype.names:
            v = row[name]
            d[name] = _s(v) if isinstance(v, (bytes, np.bytes_)) else (v.item() if hasattr(v, 'item') else v)
        out.append(d)
    return out


def net_ticks_to_datetime(ticks):
    """.NET ticks (100 ns since 0001-01-01) → datetime, or None."""
    try:
        return datetime(1, 1, 1) + timedelta(microseconds=int(ticks) // 10)
    except (TypeError, ValueError, OverflowError):
        return None


def recording_datetime(filepath):
    """Start of the recording from the file attributes, or None."""
    try:
        h5py = _h5py()
        with h5py.File(filepath, 'r') as h:
            d = h['Data'].attrs
            if 'DateInTicks' in d:
                return net_ticks_to_datetime(d['DateInTicks'])
    except (OSError, KeyError, ImportError):
        return None
    return None


def inspect(filepath):
    """Structure of the file: recordings and their streams.

    Returns a dict with 'file' (program, date, MEA layout) and 'recordings':
    a list with, per recording, 'analog' (one dict per stream: label,
    subtype, channel labels, sample rate, samples, duration), 'events',
    'segments', 'timestamps' (label and subtype).
    """
    h5py = _h5py()
    out = {'path': str(filepath), 'file': {}, 'recordings': []}
    with h5py.File(filepath, 'r') as h:
        out['protocol_version'] = int(h.attrs.get('McsHdf5ProtocolVersion', 0))
        d = h['Data']
        out['file'] = {k: _s(d.attrs[k]) for k in ('ProgramName', 'ProgramVersion', 'MeaLayout', 'MeaName', 'Date')
                       if k in d.attrs}
        out['file']['datetime'] = net_ticks_to_datetime(d.attrs['DateInTicks']) if 'DateInTicks' in d.attrs else None
        for rname in sorted(k for k in d if k.startswith('Recording_')):
            rec = d[rname]
            r = {'name': rname, 'duration_s': float(rec.attrs.get('Duration', 0)) / 1e6,
                 'analog': [], 'events': [], 'segments': [], 'timestamps': []}
            for sname in sorted(rec.get('AnalogStream', {})):
                st = rec['AnalogStream'][sname]
                info = _info_rows(st['InfoChannel'])
                tick = info[0]['Tick'] if info else None
                cd = st['ChannelData']
                r['analog'].append({'name': sname, 'label': _s(st.attrs.get('Label', '')),
                                    'subtype': _s(st.attrs.get('DataSubType', '')),
                                    'channels': [c['Label'] for c in info], 'n_channels': cd.shape[0],
                                    'sample_rate': 1e6 / tick if tick else None, 'n_samples': cd.shape[1],
                                    'duration_s': cd.shape[1] * tick / 1e6 if tick else None,
                                    'unit': info[0]['Unit'] if info else None})
            for key, field in (('EventStream', 'events'), ('SegmentStream', 'segments'), ('TimeStampStream', 'timestamps')):
                for sname in sorted(rec.get(key, {})):
                    st = rec[key][sname]
                    r[field].append({'name': sname, 'label': _s(st.attrs.get('Label', '')),
                                     'subtype': _s(st.attrs.get('DataSubType', ''))})
            out['recordings'].append(r)
    return out


def _pick_stream(rec, stream):
    """The AnalogStream group to read: by name ('Stream_0'), by (part of the)
    label, or by default the first 'Electrode' stream whose label mentions
    'Raw', else the first Electrode stream, else the first stream."""
    streams = sorted(rec['AnalogStream']) if 'AnalogStream' in rec else []
    if not streams:
        raise ValueError('no AnalogStream in the recording (only '
                         + ', '.join(k for k in rec if k.endswith('Stream')) + ')')
    if stream is not None:
        if stream in streams:
            return rec['AnalogStream'][stream]
        for s in streams:
            if str(stream).lower() in _s(rec['AnalogStream'][s].attrs.get('Label', '')).lower():
                return rec['AnalogStream'][s]
        raise ValueError(f'stream {stream!r} not found; available: '
                         + ', '.join(f"{s} ({_s(rec['AnalogStream'][s].attrs.get('Label', ''))})" for s in streams))
    electrode = [s for s in streams if _s(rec['AnalogStream'][s].attrs.get('DataSubType', '')) == 'Electrode']
    raw = [s for s in electrode if 'raw' in _s(rec['AnalogStream'][s].attrs.get('Label', '')).lower()]
    return rec['AnalogStream'][(raw or electrode or streams)[0]]


class _Decimator:
    """Block-wise FIR decimation by an integer factor q with no delay.

    Output sample m is centred on input sample m·q (the FIR is symmetric and
    the stream is padded with P/2 copies of the first sample in front and
    of the last sample at the end).  Works on (channels × time) blocks.
    """

    def __init__(self, q, fs_in, n_channels):
        from scipy.signal import firwin
        self.q = q
        taps = 36 * q + 1                              # P = 36 q (even), like the 2 kHz prototype
        self.h = firwin(taps, 0.45 * fs_in / q, fs=fs_in, window=('kaiser', 8.0))
        self.P = taps - 1
        self.buf = None                               # ext samples not yet fully consumed
        self.nch = n_channels

    def push(self, block):
        from scipy.signal import upfirdn
        block = np.asarray(block, dtype=np.float64)
        if self.buf is None:
            self.buf = np.concatenate([np.repeat(block[:, :1], self.P // 2, axis=1), block], axis=1)
        else:
            self.buf = np.concatenate([self.buf, block], axis=1)
        n_out = (self.buf.shape[1] - self.P - 1) // self.q + 1 if self.buf.shape[1] > self.P else 0
        if n_out <= 0:
            return np.empty((self.nch, 0), dtype=np.float32)
        seg = self.buf[:, :n_out * self.q + self.P]
        y = upfirdn(self.h, seg, down=self.q, axis=1)[:, self.P // self.q:self.P // self.q + n_out]
        self.buf = self.buf[:, n_out * self.q:]
        return y.astype(np.float32)

    def flush(self):
        if self.buf is None or self.buf.shape[1] == 0:
            return np.empty((self.nch, 0), dtype=np.float32)
        tail = np.repeat(self.buf[:, -1:], self.P, axis=1)
        return self.push(tail)


def load_mcs_h5(filepath, stream=None, channels=None, recording=0, max_sample_rate=MAX_SAMPLE_RATE,
                block=_BLOCK):
    """Load the electrode signal of an MCS HDF5 file.

    Parameters
    ----------
    filepath : path of the .h5 file
    stream : None, 'Stream_<s>' or part of a stream label ('Raw', 'Filter (1)');
        None picks the raw electrode stream
    channels : None (all) or list of channel labels ('E1', 'E62')
    recording : index of the recording in the file
    max_sample_rate : recordings above it are decimated to ~TARGET_SAMPLE_RATE
        (None keeps the original rate)

    Returns
    -------
    metadata : dict — sample_rate, n_samples, duration, channels (labels),
        unit, stream label, datetime, mea layout, conversion (V per ADC step
        per channel), events (list), spike_timestamps (dict label → s),
        paced (bool), original_sample_rate / decimation_factor when decimated
    df : DataFrame with 'time' (s, from 0) and one float32 column per channel
        label, values in the stream unit (volts for electrode streams)
    """
    h5py = _h5py()
    filepath = Path(filepath)
    with h5py.File(filepath, 'r') as h:
        d = h['Data']
        recs = sorted(k for k in d if k.startswith('Recording_'))
        rec = d[recs[recording]]
        st = _pick_stream(rec, stream)
        info = _info_rows(st['InfoChannel'])
        labels_all = [c['Label'] for c in info]
        if channels is None:
            sel = info
        else:
            want = [str(c) for c in channels]
            missing = [c for c in want if c not in labels_all]
            if missing:
                raise ValueError(f'channels not in the stream: {missing}; available: {labels_all}')
            sel = [info[labels_all.index(c)] for c in want]
        rows = np.array([c['RowIndex'] for c in sel], dtype=int)
        tick = float(sel[0]['Tick'])
        fs = 1e6 / tick
        scale = np.array([c['ConversionFactor'] * 10.0 ** c['Exponent'] for c in sel], dtype=np.float64)
        zero = np.array([c['ADZero'] for c in sel], dtype=np.float64)
        cd = st['ChannelData']
        n = cd.shape[1]
        ts = st['ChannelDataTimeStamps'][:]
        t0_us = float(ts[0, 0]) if ts.size else 0.0

        q = 1
        if max_sample_rate is not None and fs > max_sample_rate:
            q = int(round(fs / TARGET_SAMPLE_RATE))
        dec = _Decimator(q, fs, len(sel)) if q > 1 else None
        order = np.argsort(rows)                       # h5py wants increasing row indices
        parts = []
        for a in range(0, n, block):
            b = min(n, a + block)
            blk = cd[np.sort(rows), a:b].astype(np.float64) if len(rows) > 1 else cd[rows[0]:rows[0] + 1, a:b].astype(np.float64)
            blk = blk[np.argsort(order)]               # back to the requested channel order
            blk = (blk - zero[:, None]) * scale[:, None]
            parts.append(dec.push(blk) if dec else blk.astype(np.float32))
        if dec:
            parts.append(dec.flush())
        X = np.concatenate(parts, axis=1) if parts else np.empty((len(sel), 0), dtype=np.float32)
        n_out = -(-n // q)
        X = X[:, :n_out]

        events = _read_events(rec)
        spikes = _read_timestamps(rec)
        attrs = d.attrs
        metadata = {
            'filepath': str(filepath), 'filename': filepath.stem, 'format': 'mcs_hdf5',
            'sample_rate': fs / q, 'n_samples': int(X.shape[1]), 'duration_s': X.shape[1] * q / fs,
            'channels': [c['Label'] for c in sel], 'unit': sel[0]['Unit'],
            'stream': _s(st.attrs.get('Label', '')), 'stream_subtype': _s(st.attrs.get('DataSubType', '')),
            'recording': recs[recording], 't0_s': t0_us / 1e6,
            'conversion': {c['Label']: c['ConversionFactor'] * 10.0 ** c['Exponent'] for c in sel},
            'adc_zero': {c['Label']: c['ADZero'] for c in sel},
            'device': _s(attrs.get('MeaName', '')) or None, 'mea_layout': _s(attrs.get('MeaLayout', '')) or None,
            'program': _s(attrs.get('ProgramName', '')) or None,
            'datetime': net_ticks_to_datetime(attrs['DateInTicks']) if 'DateInTicks' in attrs else None,
            'events': events, 'spike_timestamps': spikes,
            'paced': any(e['subtype'] in STIMULUS_SUBTYPES and len(e['times_s']) > 0 for e in events),
        }
        if q > 1:
            metadata['original_sample_rate'] = fs
            metadata['decimation_factor'] = q
    df = pd.DataFrame({'time': (np.arange(X.shape[1]) * q / fs).astype(np.float64)})
    for k, c in enumerate(metadata['channels']):
        df[c] = X[k]
    return metadata, df


def _read_events(rec):
    """Event streams of a recording: one dict per entity with the times (s) and
    durations (s) of its events."""
    out = []
    if 'EventStream' not in rec:
        return out
    for sname in sorted(rec['EventStream']):
        st = rec['EventStream'][sname]
        label, sub = _s(st.attrs.get('Label', '')), _s(st.attrs.get('DataSubType', ''))
        infos = _info_rows(st['InfoEvent']) if 'InfoEvent' in st else []
        for ename in sorted(k for k in st if k.startswith('EventEntity_')):
            m = np.atleast_2d(st[ename][:])
            eid = int(ename.split('_')[-1])
            info = next((i for i in infos if i.get('EventID') == eid), {})
            out.append({'stream': sname, 'label': label, 'subtype': sub, 'entity': ename,
                        'entity_label': info.get('Label', ''), 'source_channels': info.get('SourceChannelLabels', ''),
                        'times_s': (m[0] / 1e6).tolist() if m.size else [],
                        'durations_s': (m[1] / 1e6).tolist() if m.shape[0] > 1 and m.size else []})
    return out


def _read_timestamps(rec):
    """Time stamps of the detected spikes: {channel label: [s, …]} (empty when
    the recording has no TimeStampStream)."""
    out = {}
    if 'TimeStampStream' not in rec:
        return out
    for sname in sorted(rec['TimeStampStream']):
        st = rec['TimeStampStream'][sname]
        infos = _info_rows(st['InfoTimeStamp']) if 'InfoTimeStamp' in st else []
        for ename in sorted(k for k in st if k.startswith('TimeStampEntity_')):
            tid = int(ename.split('_')[-1])
            info = next((i for i in infos if i.get('TimeStampEntityID', i.get('TimeStampID')) == tid), {})
            label = info.get('SourceChannelLabels') or info.get('Label') or f'{sname}/{ename}'
            exp = info.get('Exponent', -6)
            v = np.atleast_1d(np.asarray(st[ename][:]).ravel()).astype(np.float64)
            out[str(label).strip()] = (v * 10.0 ** float(exp)).tolist()
    return out


def read_segments(filepath, recording=0):
    """Segment streams (spike cut-outs, averages) of a recording.

    Returns one dict per segment entity: 'label', 'subtype', 'segment_id',
    'source_channel', 'pre_s', 'post_s', 'times_s' (trigger times, s) and
    'waveforms' (n segments × samples, in the source channel unit) — for
    averages 'mean' and 'std' instead of 'waveforms'.
    """
    h5py = _h5py()
    out = []
    with h5py.File(filepath, 'r') as h:
        recs = sorted(k for k in h['Data'] if k.startswith('Recording_'))
        rec = h['Data'][recs[recording]]
        if 'SegmentStream' not in rec:
            return out
        for sname in sorted(rec['SegmentStream']):
            st = rec['SegmentStream'][sname]
            label, sub = _s(st.attrs.get('Label', '')), _s(st.attrs.get('DataSubType', ''))
            segs = _info_rows(st['InfoSegment']) if 'InfoSegment' in st else []
            srcs = _info_rows(st['SourceInfoChannel']) if 'SourceInfoChannel' in st else []
            for seg in segs:
                sid = seg['SegmentID']
                src = next((c for c in srcs if c.get('ChannelID') == sid), srcs[0] if srcs else {})
                scale = src.get('ConversionFactor', 1) * 10.0 ** src.get('Exponent', 0)
                zero = src.get('ADZero', 0)
                e = {'stream': sname, 'label': label, 'subtype': sub, 'segment_id': sid,
                     'source_channel': src.get('Label', ''), 'pre_s': seg.get('PreInterval', 0) / 1e6,
                     'post_s': seg.get('PostInterval', 0) / 1e6, 'tick_s': src.get('Tick', 0) / 1e6}
                if f'SegmentData_{sid}' in st:
                    w = st[f'SegmentData_{sid}'][:].astype(np.float64)
                    e['waveforms'] = ((w - zero) * scale).T            # samples × n → n × samples
                    if f'SegmentData_ts_{sid}' in st:
                        e['times_s'] = (np.asarray(st[f'SegmentData_ts_{sid}'][:]).ravel() / 1e6).tolist()
                if f'AverageData_{sid}' in st:
                    a = st[f'AverageData_{sid}'][:].astype(np.float64)   # 2 × samples × n
                    e['mean'] = ((a[0] - zero) * scale).T
                    e['std'] = (a[1] * scale).T
                out.append(e)
    return out


def write_mcs_h5(filepath, channel_data, labels, tick_us, conversion_factor, exponent=-9, adc_zero=0,
                 unit='V', stream_label='Data Acquisition (1) Electrode Raw Data1', mea_layout='',
                 program='cardiac-fp-analyzer', start=None, events=None, compression=5):
    """Write a minimal MCS RawData HDF5 file (protocol version 3 layout).

    ``channel_data``: int array (channels × samples) of ADC codes;
    ``conversion_factor`` a scalar or one value per channel; ``events``: list of
    dicts {'label', 'subtype', 'times_s', 'durations_s'} written as
    EventStreams (one entity each). Used by the tests and to convert other
    raw formats; it writes what ``load_mcs_h5`` reads, not every field of
    the protocol.
    """
    h5py = _h5py()
    X = np.asarray(channel_data)
    if X.ndim != 2:
        raise ValueError('channel_data must be channels × samples')
    n_ch, n = X.shape
    cf = np.broadcast_to(np.asarray(conversion_factor, dtype=np.int64), (n_ch,))
    start = start or datetime.now()
    ticks = int((start - datetime(1, 1, 1)).total_seconds() * 1e7)
    str_t = h5py.string_dtype('ascii')
    info_dt = np.dtype([('ChannelID', '<i4'), ('RowIndex', '<i4'), ('GroupID', '<i4'), ('Label', str_t),
                        ('RawDataType', str_t), ('Unit', str_t), ('Exponent', '<i4'), ('ADZero', '<i4'),
                        ('Tick', '<i8'), ('ConversionFactor', '<i8'), ('ADCBits', '<i4'),
                        ('HighPassFilterType', str_t), ('HighPassFilterCutOffFrequency', str_t),
                        ('HighPassFilterOrder', '<i4'), ('LowPassFilterType', str_t),
                        ('LowPassFilterCutOffFrequency', str_t), ('LowPassFilterOrder', '<i4')])
    info = np.zeros(n_ch, dtype=info_dt)
    for k in range(n_ch):
        info[k] = (k, k, 0, str(labels[k]), 'Int', unit, int(exponent), int(adc_zero), int(tick_us), int(cf[k]),
                   16 if X.dtype.itemsize <= 2 else 24, '', '-1', -1, '', '-1', -1)
    with h5py.File(filepath, 'w') as h:
        h.attrs['McsHdf5ProtocolType'] = np.bytes_(b'RawData')
        h.attrs['McsHdf5ProtocolVersion'] = np.int32(3)
        h.attrs['GeneratingApplicationName'] = np.bytes_(program.encode('ascii', 'replace'))
        d = h.create_group('Data')
        for k, v in (('ProgramName', program), ('ProgramVersion', ''), ('MeaLayout', mea_layout), ('MeaName', mea_layout),
                     ('MeaSN', ''), ('Comment', ''), ('Date', start.strftime('%Y-%m-%d %H:%M:%S')), ('FileGUID', '')):
            d.attrs[k] = np.bytes_(str(v).encode('ascii', 'replace'))
        d.attrs['DateInTicks'] = np.int64(ticks)
        rec = d.create_group('Recording_0')
        rec.attrs['Duration'] = np.int64(n * tick_us)
        rec.attrs['TimeStamp'] = np.int64(0)
        rec.attrs['RecordingID'] = np.int32(0)
        for k in ('Comment', 'Label', 'RecordingType'):
            rec.attrs[k] = np.bytes_(b'')
        st = rec.create_group('AnalogStream').create_group('Stream_0')
        st.attrs['Label'] = np.bytes_(stream_label.encode('ascii', 'replace'))
        st.attrs['DataSubType'] = np.bytes_(b'Electrode')
        st.attrs['StreamType'] = np.bytes_(b'Analog')
        st.attrs['StreamInfoVersion'] = np.int32(1)
        st.attrs['StreamGUID'] = np.bytes_(b'')
        st.attrs['SourceStreamGUID'] = np.bytes_(b'')
        st.create_dataset('ChannelData', data=X.astype(np.int32), chunks=(min(n_ch, 64), min(n, 2048)),
                          compression='gzip', compression_opts=compression, shuffle=True)
        st.create_dataset('ChannelDataTimeStamps', data=np.array([[0, 0, n - 1]], dtype=np.int64))
        ds = st.create_dataset('InfoChannel', data=info)
        ds.attrs['InfoVersion'] = np.int32(1)
        if events:
            es = rec.create_group('EventStream')
            ev_dt = np.dtype([('EventID', '<i4'), ('GroupID', '<i4'), ('Label', str_t), ('RawDataType', str_t),
                              ('RawDataBytes', '<i4'), ('SourceChannelIDs', str_t), ('SourceChannelLabels', str_t)])
            for i, e in enumerate(events):
                g = es.create_group(f'Stream_{i}')
                g.attrs['Label'] = np.bytes_(str(e.get('label', '')).encode('ascii', 'replace'))
                g.attrs['DataSubType'] = np.bytes_(str(e.get('subtype', 'DigitalPort')).encode('ascii'))
                g.attrs['StreamType'] = np.bytes_(b'Event')
                g.attrs['StreamInfoVersion'] = np.int32(1)
                t = np.asarray(e.get('times_s', []), dtype=np.float64)
                dur = np.asarray(e.get('durations_s', np.zeros_like(t)), dtype=np.float64)
                m = np.zeros((5, len(t)), dtype=np.int64)
                m[0] = np.round(t * 1e6)
                m[1] = np.round(dur * 1e6)
                g.create_dataset('EventEntity_0', data=m)
                ie = np.zeros(1, dtype=ev_dt)
                ie[0] = (0, 0, str(e.get('label', '')), 'Int', 4, '', '')
                g.create_dataset('InfoEvent', data=ie)
    return Path(filepath)
