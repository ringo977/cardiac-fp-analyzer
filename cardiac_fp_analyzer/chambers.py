"""
chambers.py — Electrode layouts of multi-chamber chips.

A Multi Channel Systems file holds every electrode of the chip; a chip
holds several microtissues, one per chamber. The pipeline measures one
tissue at a time, so a multi-chamber recording is analysed as one
recording per chamber, each restricted to the chamber's electrodes
(``analyze.find_recordings`` / ``sample_sheet.plan_batch``).

Layouts
-------
``uheart_mvp_64`` — µHeart MVP, 64 channels (PHOENIX deliverable D1.2,
confirmed by the lab, Oct 2026). Four modules of 16 electrodes: two
stimulation pairs at the ends of the tissue channel (large electrodes: the
1st–2nd and 14th–15th channel of each block of 15) and 12 recording
electrodes of 30 µm in a line along the channel, pitch 400 µm, the 12th
being one of the four extra pads E61–E64 that the acquisition software
appends after E60. Chamber letters are the lab's:

    A = E16–E30 + E62    B = E31–E45 + E63    C = E1–E15 + E61    D = E46–E60 + E64

The position of the extra pad along the row (after the 5th or 6th
recording electrode) was found from the activation order of the beats and
matters only for conduction velocity.

A layout is picked by name (``AnalysisConfig.chamber_layout``), or
automatically from the channel labels of the file ('auto': a 64-channel MCS
file with E1…E64 is a µHeart MVP). 'none' analyses the file as one tissue.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chamber:
    name: str
    electrodes: tuple          # every electrode of the chamber, file order
    stimulation: tuple         # stimulation electrodes (not used for measurements)
    row: tuple                 # recording electrodes in their physical order along the tissue
    pitch_mm: float            # distance between neighbouring recording electrodes

    @property
    def recording(self):
        return tuple(e for e in self.electrodes if e not in self.stimulation)


@dataclass(frozen=True)
class Layout:
    name: str
    description: str
    chambers: tuple            # of Chamber
    channels: tuple = field(default=())   # channel labels the layout expects (for detection)

    def chamber_of(self, electrode):
        for c in self.chambers:
            if electrode in c.electrodes:
                return c
        return None

    def __getitem__(self, name):
        for c in self.chambers:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def names(self):
        return tuple(c.name for c in self.chambers)


def _uheart_chamber(name, first, pad, pad_after, pitch=0.400):
    block = [f'E{k}' for k in range(first, first + 15)]
    stim = (block[0], block[1], block[13], block[14])
    rec = block[2:13]                      # 11 recording electrodes of the block
    row = tuple(rec[:pad_after] + [pad] + rec[pad_after:])
    return Chamber(name, tuple(block + [pad]), stim, row, pitch)


UHEART_MVP_64 = Layout(
    name='uheart_mvp_64',
    description='µHeart MVP, 64 channels: 4 chambers × (12 recording + 4 stimulation electrodes), pitch 400 µm',
    chambers=(
        _uheart_chamber('A', 16, 'E62', 6),
        _uheart_chamber('B', 31, 'E63', 5),
        _uheart_chamber('C', 1, 'E61', 6),
        _uheart_chamber('D', 46, 'E64', 6),
    ),
    channels=tuple(f'E{k}' for k in range(1, 65)),
)

LAYOUTS = {UHEART_MVP_64.name: UHEART_MVP_64}


def detect_layout(channels):
    """Layout whose channel set matches the file's labels, or None."""
    chs = set(map(str, channels))
    for lay in LAYOUTS.values():
        if lay.channels and set(lay.channels) == chs:
            return lay
    return None


def layout_for(channels, choice='auto'):
    """Layout to use for a recording with these channel labels.

    ``choice``: 'auto' (detect from the labels, None if no layout matches),
    'none' (single tissue), or a layout name (ValueError if unknown).
    """
    if choice in (None, 'none', ''):
        return None
    if choice == 'auto':
        return detect_layout(channels)
    if choice not in LAYOUTS:
        raise ValueError(f'unknown chamber layout {choice!r}; known: {sorted(LAYOUTS)}')
    return LAYOUTS[choice]
