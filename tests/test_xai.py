"""Tests for the explainable-AI layer and the report renderer."""

from __future__ import annotations

import io

import numpy as np
import pytest

from app.data.synthetic import synth_ecg, synth_eeg, synth_xray
from app.ml.xai import gradcam, plots


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def test_plot_signal_returns_base64_png():
    signal, _ = synth_ecg("NORM", duration_s=5, seed=3)
    encoded = plots.plot_signal(signal, 500.0, title="test")
    assert isinstance(encoded, str) and len(encoded) > 100
    import base64

    assert base64.b64decode(encoded)[:4] == b"\x89PNG"


def test_plot_saliency_handles_mismatched_lengths():
    signal, _ = synth_ecg("NORM", duration_s=3, seed=3)
    encoded = plots.plot_saliency(signal, np.linspace(0, 1, 100), 500.0)
    assert isinstance(encoded, str) and len(encoded) > 100


def test_plot_overlay_and_probabilities():
    image = synth_xray("NORMAL", size=64, seed=5)
    cam = np.random.default_rng(0).random((64, 64))
    assert len(plots.plot_overlay(image, cam)) > 100
    assert len(plots.plot_class_probabilities({"NORMAL": 0.8, "PNEUMONIA": 0.2})) > 100


def test_plot_feature_contributions():
    assert len(plots.plot_feature_contributions([("heart rate", 0.5), ("ST", -0.2)])) > 100


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def test_grad_cam_runs_on_the_ecg_network():
    import torch

    from app.ml.ecg.model import ECGNet
    from app.ml.ecg.preprocess import build_model_input, preprocess_ecg
    from app.ml.ecg.features import extract_features, feature_vector

    signal, fs = synth_ecg("NORM", duration_s=10, seed=11)
    pre = preprocess_ecg(signal, fs)
    features = extract_features(pre)

    network = ECGNet()
    network.eval()
    window = build_model_input(pre.window[0])
    fvec = feature_vector(features)
    cam = gradcam.grad_cam_for_ecg(network, window, fvec, 0, "cpu")
    assert cam.shape == (2500,)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0 + 1e-6


def test_grad_cam_runs_on_the_xray_network():
    from app.ml.xray.features import feature_vector
    from app.ml.xray.model import XrayCNN
    from app.ml.xray.preprocess import build_model_input, preprocess_xray_array

    image = synth_xray("NORMAL", seed=13)
    pre = preprocess_xray_array(image)
    features = __import__("app.ml.xray.features", fromlist=["extract_features"]).extract_features(pre)

    network = XrayCNN()
    network.eval()
    cam = gradcam.grad_cam_for_xray(
        network, build_model_input(pre.grayscale), feature_vector(features), 0, "cpu"
    )
    assert cam.shape == (224, 224)
    assert float(cam.max()) <= 1.0 + 1e-6


def test_integrated_gradients_returns_an_attribution_map():
    import torch

    from app.ml.ecg.model import ECGNet
    from app.ml.ecg.preprocess import build_model_input, preprocess_ecg
    from app.ml.ecg.features import extract_features, feature_vector

    signal, fs = synth_ecg("NORM", duration_s=10, seed=13)
    pre = preprocess_ecg(signal, fs)
    features = extract_features(pre)

    network = ECGNet()
    network.eval()
    # ECGNet consumes (batch, channels, samples); the network input is the
    # conditioned model window, not the whole recording.
    window = torch.from_numpy(build_model_input(pre.window[0])).float().unsqueeze(0)
    fvec = torch.from_numpy(feature_vector(features)).float().unsqueeze(0)
    attribution = gradcam.integrated_gradients(network, window, fvec, 0, steps=4)
    assert attribution.shape == (window.shape[-1],)


def test_feature_attributions_rank_the_inputs():
    import torch

    from app.ml.ecg.model import ECGNet
    from app.ml.ecg.preprocess import build_model_input, preprocess_ecg
    from app.ml.ecg.features import FEATURE_ORDER, extract_features, feature_vector

    signal, fs = synth_ecg("NORM", duration_s=10, seed=17)
    pre = preprocess_ecg(signal, fs)
    features = extract_features(pre)

    network = ECGNet()
    network.eval()
    window = torch.from_numpy(build_model_input(pre.window[0])).float().unsqueeze(0)
    fvec = torch.from_numpy(feature_vector(features)).float().unsqueeze(0)
    baseline = torch.zeros_like(fvec)
    attributions = gradcam.feature_attributions(network, window, fvec, 0, baseline)
    assert attributions.shape == (len(FEATURE_ORDER),)


def test_occlusion_sensitivity_returns_segments():
    from app.ml.xai.gradcam import occlusion_sensitivity

    signal = np.sin(np.linspace(0, 30, 1500)).astype(np.float32)

    def score(modified: np.ndarray) -> float:
        return float(np.mean(np.abs(modified)))

    sensitivity, segments = occlusion_sensitivity(score, signal, window_size=100, stride=50)
    assert sensitivity.shape == signal.shape
    assert len(segments) > 0
    assert all("importance" in segment for segment in segments)
