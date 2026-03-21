"""
cardiac_fp_analyzer — Field Potential Analysis for hiPSC-CM µECG recordings.

Modules:
  config          — Central configuration (AnalysisConfig, presets, JSON I/O)
  loader          — CSV parser for Digilent WaveForms files
  filtering       — Signal conditioning (notch, bandpass, smoothing)
  plotting        — Smart downsampled visualization
  beat_detection  — Depolarization spike detection and segmentation
  parameters      — Electrophysiology parameter extraction (BP, FPD, FPDc, etc.)
  quality_control — Signal quality grading and beat validation
  arrhythmia      — Arrhythmia detection and classification
  normalization   — Baseline normalization and TdP risk scoring
  cessation       — Beating cessation / quiescence detection
  spectral        — Frequency domain analysis (PSD, entropy, harmonics)
  report          — Excel and PDF report generation
  analyze         — Main pipeline orchestrator (single file + batch)
"""

__version__ = '3.0.0'
