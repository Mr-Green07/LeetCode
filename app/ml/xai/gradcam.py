"""Gradient-based explainability for 2-D and 1-D networks.

A single implementation covers all three modalities:

* X-ray  -> Grad-CAM over the last convolutional block (2-D activation map);
* ECG    -> Grad-CAM1D over the final residual block (1-D activation);
* EEG    -> Grad-CAM over the last EEGNet convolutional block.

The routine works on any convolutional target layer by capturing the layer's
forward activation and backward gradient.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _spatial_dims(activation: "torch.Tensor") -> Tuple[int, ...]:
    """Indices of the spatial dimensions of an activation tensor."""
    if activation.dim() == 2:  # (B, C) - no spatial axis
        return ()
    return tuple(range(2, activation.dim()))


def grad_cam(
    network: Any,
    input_tensor: "torch.Tensor",
    feature_tensor: Optional["torch.Tensor"],
    target_class: int,
    target_layer: Any,
) -> np.ndarray:
    """Compute a Grad-CAM map for ``target_class``.

    Returns a numpy array whose shape matches the *spatial* dimensions of
    ``input_tensor`` (e.g. ``(H, W)`` for images, ``(T,)`` for signals).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)

    activations: Dict[str, Any] = {}
    gradients: Dict[str, Any] = {}

    def _forward_hook(_module, _inputs, output):
        activations["value"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_forward = target_layer.register_forward_hook(_forward_hook)
    handle_backward = target_layer.register_full_backward_hook(_backward_hook)

    try:
        network.eval()
        network.zero_grad(set_to_none=True)
        if feature_tensor is None:
            logits = network(input_tensor)
        else:
            logits = network(input_tensor, feature_tensor)

        score = logits[0, target_class]
        score.backward()

        if "value" not in activations or "value" not in gradients:
            logger.warning("Grad-CAM hooks did not fire - returning an empty map")
            return np.zeros(1, dtype=np.float32)

        activation = activations["value"]
        gradient = gradients["value"]

        spatial = _spatial_dims(activation)
        if not spatial:
            return np.zeros(1, dtype=np.float32)

        weights = gradient.mean(dim=spatial)  # (C,)
        shape = (1, -1) + (1,) * len(spatial)
        cam = (weights.reshape(shape) * activation).sum(dim=1)  # (1, *spatial)
        cam = F.relu(cam)

        target_size = tuple(input_tensor.shape[2:]) if input_tensor.dim() > 2 else tuple(input_tensor.shape[1:])
        if tuple(cam.shape[1:]) != target_size:
            # 1-D activations need "linear" (bilinear requires a 4-D tensor),
            # 2-D activations use "bilinear" to reach the original resolution.
            mode = "linear" if len(target_size) == 1 else "bilinear"
            cam = F.interpolate(
                cam.unsqueeze(1), size=target_size, mode=mode, align_corners=False
            ).squeeze(1)

        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam = np.squeeze(cam)
        minimum, maximum = float(cam.min()), float(cam.max())
        if maximum - minimum < 1e-9:
            return np.zeros_like(cam)
        return (cam - minimum) / (maximum - minimum)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Grad-CAM failed: %s", exc)
        return np.zeros(1, dtype=np.float32)
    finally:
        handle_forward.remove()
        handle_backward.remove()


def grad_cam_for_ecg(network: Any, window: np.ndarray, features: np.ndarray, target_class: int, device: str) -> np.ndarray:
    """Grad-CAM1D for the ECG ResNet1D network."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)
    target_layer = network.stage3[-1].conv2
    signal = torch.from_numpy(window).float().unsqueeze(0).to(device)
    feat = torch.from_numpy(features).float().unsqueeze(0).to(device)
    return grad_cam(network, signal, feat, target_class, target_layer)


def grad_cam_for_eeg(network: Any, window: np.ndarray, features: np.ndarray, target_class: int, device: str) -> np.ndarray:
    """Grad-CAM for the EEG EEGNet network."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)
    target_layer = network.block3[0]
    signal = torch.from_numpy(window).float().unsqueeze(0).to(device)
    feat = torch.from_numpy(features).float().unsqueeze(0).to(device)
    return grad_cam(network, signal, feat, target_class, target_layer)


def grad_cam_for_xray(network: Any, image: np.ndarray, features: np.ndarray, target_class: int, device: str) -> np.ndarray:
    """Grad-CAM for the chest X-ray CNN."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)
    target_layer = network.block3.conv
    tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    feat = torch.from_numpy(features).float().unsqueeze(0).to(device)
    return grad_cam(network, tensor, feat, target_class, target_layer)


def integrated_gradients(
    network: Any,
    input_tensor: "torch.Tensor",
    feature_tensor: Optional["torch.Tensor"],
    target_class: int,
    baseline: Optional["torch.Tensor"] = None,
    steps: int = 32,
) -> np.ndarray:
    """Integrated gradients attribution for the model input."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    total_grad = torch.zeros_like(input_tensor)
    alphas = torch.linspace(1.0 / steps, 1.0, steps)

    for alpha in alphas:
        interpolated = (baseline + alpha * (input_tensor - baseline)).clone().requires_grad_(True)
        if feature_tensor is None:
            logits = network(interpolated)
        else:
            logits = network(interpolated, feature_tensor)
        score = logits[0, target_class]
        network.zero_grad(set_to_none=True)
        score.backward()
        total_grad = total_grad + interpolated.grad.detach()

    average_grad = total_grad / steps
    attribution = (input_tensor.detach() - baseline) * average_grad
    result = attribution.detach().cpu().numpy().astype(np.float32)
    return np.squeeze(result)


def occlusion_sensitivity(
    score_fn: Callable[[np.ndarray], float],
    signal: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Occlusion sensitivity for 1-D signals.

    ``score_fn`` receives a modified signal and returns the score of the
    positive class.  A drop in score marks the region as causally relevant.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    baseline_score = float(score_fn(signal))
    sensitivity = np.zeros_like(signal, dtype=np.float32)

    segments: List[Dict[str, float]] = []
    if signal.size <= window_size:
        return sensitivity, segments

    for start in range(0, signal.size - window_size + 1, stride):
        end = start + window_size
        modified = signal.copy()
        modified[start:end] = float(np.mean(signal))
        score = float(score_fn(modified))
        drop = baseline_score - score
        sensitivity[start:end] = drop
        segments.append({"start": float(start), "end": float(end), "importance": float(drop)})

    maximum = float(sensitivity.max())
    minimum = float(sensitivity.min())
    if maximum - minimum > 1e-9:
        sensitivity = (sensitivity - minimum) / (maximum - minimum)
    else:
        sensitivity = np.zeros_like(sensitivity)

    segments.sort(key=lambda item: item["importance"], reverse=True)
    return sensitivity, segments


def feature_attributions(
    network: Any,
    input_tensor: "torch.Tensor",
    feature_tensor: Optional["torch.Tensor"],
    target_class: int,
    baseline_features: Optional["torch.Tensor"] = None,
) -> np.ndarray:
    """Gradient-based attribution of each clinical feature to the score.

    Each feature is perturbed to its baseline (0) while the others stay fixed;
    the resulting change in the target score is the feature's attribution.
    """
    if not TORCH_AVAILABLE or feature_tensor is None:  # pragma: no cover
        return np.zeros(1, dtype=np.float32)

    network.eval()
    with torch.no_grad():
        base_logits = network(input_tensor, feature_tensor)
        base_score = float(base_logits[0, target_class])

    attributions = np.zeros(feature_tensor.shape[1], dtype=np.float32)
    for index in range(feature_tensor.shape[1]):
        perturbed = feature_tensor.clone()
        perturbed[0, index] = baseline_features[0, index] if baseline_features is not None else 0.0
        with torch.no_grad():
            logits = network(input_tensor, perturbed)
        attributions[index] = base_score - float(logits[0, target_class])

    magnitude = float(np.abs(attributions).max())
    if magnitude > 1e-9:
        attributions = attributions / magnitude
    return attributions
