"""
cardiac_fp_analyzer — Field Potential Analysis for hiPSC-CM µECG recordings.

Modules:
  loader        — CSV parser for Digilent WaveForms files
  filtering     — Signal conditioning (notch, bandpass, smoothing)
  plotting      — Smart downsampled visualization
  beat_detection — Depolarization spike detection and segmentation
  parameters    — Electrophysiology parameter extraction (BP, FPD, FPDc, etc.)
  arrhythmia    — Arrhythmia detection and classification
  report        — Excel and PDF report generation
"""

__version__ = '1.0.0'
