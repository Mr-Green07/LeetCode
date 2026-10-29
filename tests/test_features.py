"""Tests for the signal conditioning and feature extraction layers."""

from __future__ import annotations

import numpy as np
import pytest

from app.data.synthetic import synth_ecg, synth_eeg, synth_xray
from app.ml.ecg.features import extract_features as ecg_features
from app.ml.ecg.preprocess import (
    bandpass,
    detect_r_peaks,
    notch_filter,
    preprocess_ecg,
    resample_signal,
    zscore,
)
from app.ml.eeg.features import extract_features as eeg_features
from app.ml.eeg.preprocess import preprocess_eeg
from app.ml.xray.features import extract_features as xray_features
from app.ml.xray.preprocess import clahe, preprocess_xray_array


# --------------------------------------------------------------------------- #
# Generic signal utilities
# --------------------------------------------------------------------------- #
def test_bandpass_removes_out_of_band_energy():
    fs = 500.0
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
    filtered = bandpass(signal, fs, 1.0, 20.0)
    assert float(np.std(filtered)) < float(np.std(signal))


def test_notch_filter_leaves_signal_intact():
    fs = 500.0
    t = np.arange(0, 3, 1 / fs)
    signal = np.sin(2 * np.pi * 7 * t)
    filtered = notch_filter(signal, fs, 50.0)
    assert float(np.std(filtered)) > 0.5 * float(np.std(signal))


def test_resample_changes_the_length():
    signal = np.sin(np.linspace(0, 20, 1000))
    resampled = resample_signal(signal, 500.0, 250.0)
    assert resampled.size == pytest.approx(500, rel=0.05)


def test_zscore_is_standardised():
    normalised = zscore(np.random.default_rng(0).normal(5.0, 2.0, 500))
    assert float(np.mean(normalised)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.std(normalised)) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# ECG
# --------------------------------------------------------------------------- #
def test_r_peak_detection_finds_every_beat():
    signal, fs = synth_ecg("NORM", duration_s=10, seed=7)
    peaks = detect_r_peaks(signal, fs)
    # ~68 bpm over 10 seconds -> 11-12 beats
    assert 10 <= peaks.size <= 14


def test_ecg_preprocessing_shapes_and_quality():
    signal, fs = synth_ecg("NORM", duration_s=10, seed=7)
    pre = preprocess_ecg(signal, fs)
    assert pre.window.shape == (1, 2500)
    assert pre.clean.shape[0] == 1
    assert pre.heart_rate is not None
    assert 55 < pre.heart_rate < 85
    assert 0.0 <= pre.quality["sqi"] <= 1.0


def test_ecg_features_separate_the_classes():
    expected = {
        "NORM": lambda f: f["st_deviation"].value == pytest.approx(0.0, abs=0.05),
        "MI": lambda f: f["st_deviation"].value < -0.03,
        "STDE": lambda f: f["st_deviation"].value > 0.05,
        "AFIB": lambda f: f["rr_irregularity"].value > 5.0,
        "PVC": lambda f: f["pvc_fraction"].value > 10.0,
    }
    for label, check in expected.items():
        signal, fs = synth_ecg(label, duration_s=10, seed=11)
        features = ecg_features(preprocess_ecg(signal, fs))
        assert check(features), f"{label} did not produce the expected features"


def test_ecg_features_have_a_complete_reference_table():
    signal, fs = synth_ecg("NORM", duration_s=10, seed=5)
    features = ecg_features(preprocess_ecg(signal, fs))
    assert len(features) >= 15
    for feature in features.values():
        assert feature.unit
        assert feature.explanation
        assert feature.status in {"normal", "borderline", "abnormal"}


# --------------------------------------------------------------------------- #
# EEG
# --------------------------------------------------------------------------- #
def test_eeg_preprocessing_shapes():
    signal, fs = synth_eeg("NORMAL", duration_s=10, seed=3)
    pre = preprocess_eeg(signal, fs)
    assert pre.window.shape == (1, 2560)
    assert pre.clean.shape[0] == 1
    assert 0.0 <= pre.quality["sqi"] <= 1.0


def test_eeg_band_powers_separate_the_classes():
    normal, _ = synth_eeg("NORMAL", duration_s=10, seed=9)
    parkinson, _ = synth_eeg("PARKINSON", duration_s=10, seed=9)
    depression, _ = synth_eeg("DEPRESSION", duration_s=10, seed=9)

    normal_features = eeg_features(preprocess_eeg(normal, 256.0))
    parkinson_features = eeg_features(preprocess_eeg(parkinson, 256.0))
    depression_features = eeg_features(preprocess_eeg(depression, 256.0))

    assert parkinson_features["beta_power"].value > normal_features["beta_power"].value * 1.5
    assert parkinson_features["alpha_power"].value < normal_features["alpha_power"].value * 0.5
    assert depression_features["theta_beta_ratio"].value > normal_features["theta_beta_ratio"].value * 2.0


def test_eeg_artifact_ratio_is_low_for_clean_data():
    signal, fs = synth_eeg("NORMAL", duration_s=10, seed=13)
    pre = preprocess_eeg(signal, fs)
    assert pre.artifact_ratio < 0.2


# --------------------------------------------------------------------------- #
# X-ray
# --------------------------------------------------------------------------- #
def test_clahe_preserves_dimensions():
    image = (np.random.default_rng(0).normal(128, 30, (64, 64))).clip(0, 255).astype(np.uint8)
    enhanced = clahe(image)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_xray_preprocessing_and_features():
    image = synth_xray("NORMAL", seed=17)
    pre = preprocess_xray_array(image)
    assert pre.array.shape == (1, 224, 224)
    assert 0.0 <= pre.array.min() and pre.array.max() <= 1.0

    features = xray_features(pre)
    assert len(features) == 11
    assert features["image_quality"].value > 0.0


def test_xray_regional_features_detect_the_lesion():
    normal = xray_features(preprocess_xray_array(synth_xray("NORMAL", seed=21)))
    tuberculosis = xray_features(preprocess_xray_array(synth_xray("TUBERCULOSIS", seed=21)))
    covid = xray_features(preprocess_xray_array(synth_xray("COVID19", seed=21)))
    effusion = xray_features(preprocess_xray_array(synth_xray("EFFUSION", seed=21)))

    assert tuberculosis["apical_opacity"].value > normal["apical_opacity"].value * 2
    assert covid["peripheral_infiltration"].value > normal["peripheral_infiltration"].value * 1.5
    assert effusion["costophrenic_blunting"].value > normal["costophrenic_blunting"].value * 2
