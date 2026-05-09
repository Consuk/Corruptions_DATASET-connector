from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEPTH_EXTENSIONS = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

CORRUPTION_ORDER = [
    "brightness",
    "bright",
    "dark",
    "fog",
    "frost",
    "snow",
    "contrast",
    "defocus_blur",
    "defocus",
    "glass_blur",
    "glass",
    "motion_blur",
    "motion",
    "zoom_blur",
    "zoom",
    "elastic_transform",
    "elastic",
    "quantization",
    "quant",
    "gaussian_noise",
    "gaussian",
    "impulse_noise",
    "impulse",
    "shot_noise",
    "shot",
    "iso_noise",
    "iso",
    "pixelate",
    "jpeg_compression",
    "jpeg",
]

DISPLAY_LABELS = {
    "brightness": "Bright",
    "bright": "Bright",
    "dark": "Dark",
    "fog": "Fog",
    "frost": "Frost",
    "snow": "Snow",
    "contrast": "Contrast",
    "defocus_blur": "Defocus",
    "defocus": "Defocus",
    "glass_blur": "Glass",
    "glass": "Glass",
    "motion_blur": "Motion",
    "motion": "Motion",
    "zoom_blur": "Zoom",
    "zoom": "Zoom",
    "elastic_transform": "Elastic",
    "elastic": "Elastic",
    "quantization": "Quant",
    "quant": "Quant",
    "gaussian_noise": "Gaussian",
    "gaussian": "Gaussian",
    "impulse_noise": "Impulse",
    "impulse": "Impulse",
    "shot_noise": "Shot",
    "shot": "Shot",
    "iso_noise": "ISO",
    "iso": "ISO",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG",
    "jpeg": "JPEG",
}


@dataclass
class ModelSpec:
    name: str
    kind: str
    path: Path
    code_root: Optional[Path] = None
    checkpoint_files: Optional[list[str]] = None
    pretrained_path: Optional[Path] = None
    load_audit: Optional[list[dict]] = None


def prepare_import_path(code_root: Optional[Path], purge_prefixes: Iterable[str]) -> None:
    if code_root is not None:
        code_root = code_root.expanduser().resolve()
        root_str = str(code_root)
        sys.path[:] = [p for p in sys.path if p != root_str]
        sys.path.insert(0, root_str)

    for prefix in purge_prefixes:
        for module_name in list(sys.modules):
            if module_name == prefix or module_name.startswith(prefix + "."):
                del sys.modules[module_name]


class Monodepth2Predictor:
    def __init__(
        self,
        name: str,
        weights_folder: Path,
        code_root: Optional[Path],
        num_layers: int,
        height: int,
        width: int,
        min_depth: float,
        max_depth: float,
        device: str,
        output_mode: str,
    ) -> None:
        self.name = name
        self.weights_folder = weights_folder
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device_name = device
        self.output_mode = output_mode
        self.load_audit = []

        prepare_import_path(code_root, purge_prefixes=["networks", "layers"])

        try:
            import torch
            import networks
            from layers import disp_to_depth
        except Exception as exc:  # pragma: no cover - depends on the VM repo.
            raise RuntimeError(
                "Could not import torch/networks/layers. Pass --code_root pointing "
                "to the repo that contains networks.py and layers.py."
            ) from exc

        self.torch = torch
        self.disp_to_depth = disp_to_depth

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if not weights_folder.is_dir():
            raise FileNotFoundError(f"{name}: weights folder not found: {weights_folder}")

        encoder_path = weights_folder / "encoder.pth"
        decoder_path = weights_folder / "depth.pth"
        if not encoder_path.is_file() or not decoder_path.is_file():
            raise FileNotFoundError(
                f"{name}: expected encoder.pth and depth.pth inside {weights_folder}"
            )

        self.encoder = networks.ResnetEncoder(num_layers, False)
        self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(4))

        encoder_dict = self._load_torch_file(encoder_path)
        decoder_dict = self._load_torch_file(decoder_path)

        encoder_dict = self._unwrap_state_dict(encoder_dict)
        decoder_dict = self._unwrap_state_dict(decoder_dict)
        if isinstance(encoder_dict, dict):
            self.height = int(encoder_dict.get("height", self.height))
            self.width = int(encoder_dict.get("width", self.width))

        self._load_partial(self.encoder, encoder_dict, model_name=f"{name}/encoder")
        self._load_partial(self.depth_decoder, decoder_dict, model_name=f"{name}/depth")

        self.encoder.to(self.device).eval()
        self.depth_decoder.to(self.device).eval()

    def _load_torch_file(self, path: Path):
        try:
            return self.torch.load(str(path), map_location=self.device, weights_only=False)
        except TypeError:
            return self.torch.load(str(path), map_location=self.device)

    @staticmethod
    def _unwrap_state_dict(obj):
        if isinstance(obj, dict):
            for key in ("state_dict", "model", "net", "encoder", "depth_decoder"):
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
        return obj

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        out = {}
        for key, value in state_dict.items():
            out[key[7:] if key.startswith("module.") else key] = value
        return out

    @staticmethod
    def _candidate_keys(key: str) -> list[str]:
        keys = [key]
        prefixes = [
            "module.",
            "model.",
            "net.",
            "encoder.",
            "depth.",
            "depth_decoder.",
            "module.model.",
            "module.net.",
            "module.encoder.",
            "module.depth.",
            "module.depth_decoder.",
        ]
        for prefix in prefixes:
            if key.startswith(prefix):
                keys.append(key[len(prefix) :])

        parts = key.split(".")
        for drop in range(1, min(4, len(parts))):
            keys.append(".".join(parts[drop:]))

        out = []
        for candidate in keys:
            if candidate and candidate not in out:
                out.append(candidate)
        return out

    def _match_state_dict(self, model, state_dict, model_name: str):
        state_dict = self._strip_module_prefix(state_dict)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"{model_name}: checkpoint is not a state dict")
        model_dict = model.state_dict()
        filtered = {}
        source_matched = set()
        shape_mismatches = []
        for key, value in state_dict.items():
            key_had_shape_mismatch = False
            for candidate in self._candidate_keys(key):
                if candidate not in model_dict:
                    continue
                if hasattr(value, "shape") and hasattr(model_dict[candidate], "shape"):
                    if tuple(value.shape) != tuple(model_dict[candidate].shape):
                        key_had_shape_mismatch = True
                        if len(shape_mismatches) < 12:
                            shape_mismatches.append(
                                {
                                    "checkpoint_key": key,
                                    "model_key": candidate,
                                    "checkpoint_shape": list(value.shape),
                                    "model_shape": list(model_dict[candidate].shape),
                                }
                            )
                        continue
                filtered[candidate] = value
                source_matched.add(key)
                break
            if key_had_shape_mismatch:
                continue
        if not filtered:
            raise RuntimeError(f"{model_name}: no checkpoint keys matched the model")
        missing_keys = [key for key in model_dict.keys() if key not in filtered]
        unexpected_keys = [k for k in state_dict.keys() if k not in source_matched]
        audit = {
            "module": model_name,
            "class": model.__class__.__name__,
            "checkpoint_keys": len(state_dict),
            "model_keys": len(model_dict),
            "loaded_keys": len(filtered),
            "missing_model_keys": len(missing_keys),
            "unexpected_checkpoint_keys": len(unexpected_keys),
            "shape_mismatch_count": len(shape_mismatches),
            "missing_model_keys_sample": missing_keys[:12],
            "unexpected_checkpoint_keys_sample": unexpected_keys[:12],
            "shape_mismatches_sample": shape_mismatches,
        }
        return filtered, audit

    def _load_partial(self, model, state_dict, model_name: str) -> None:
        filtered, audit = self._match_state_dict(model, state_dict, model_name)
        result = model.load_state_dict(filtered, strict=False)
        audit["missing_model_keys_sample"] = list(getattr(result, "missing_keys", []))[:12]
        audit["unexpected_checkpoint_keys_sample"] = list(getattr(result, "unexpected_keys", []))[:12]
        if not hasattr(self, "load_audit"):
            self.load_audit = []
        self.load_audit.append(audit)

    def predict(self, rgb: Image.Image) -> np.ndarray:
        image = rgb.convert("RGB").resize((self.width, self.height), Image.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            features = self.encoder(tensor)
            output = self.depth_decoder(features)
            scaled_disp, depth = self.disp_to_depth(
                output[("disp", 0)], self.min_depth, self.max_depth
            )
            if self.output_mode == "depth":
                pred = depth[0, 0].detach().cpu().numpy()
            else:
                pred = scaled_disp[0, 0].detach().cpu().numpy()

        pred_img = Image.fromarray(pred.astype(np.float32), mode="F")
        pred_img = pred_img.resize(rgb.size, Image.BILINEAR)
        return np.asarray(pred_img).astype(np.float32)


class MonoViTPredictor(Monodepth2Predictor):
    def __init__(
        self,
        name: str,
        weights_folder: Path,
        code_root: Optional[Path],
        height: int,
        width: int,
        min_depth: float,
        max_depth: float,
        device: str,
        output_mode: str,
    ) -> None:
        self.name = name
        self.weights_folder = weights_folder
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device_name = device
        self.output_mode = output_mode
        self.load_audit = []

        prepare_import_path(code_root, purge_prefixes=["networks", "layers"])

        try:
            import torch
            import networks
            from layers import disp_to_depth
        except Exception as exc:  # pragma: no cover - depends on the VM repo.
            raise RuntimeError(
                "Could not import MonoViT networks/layers. Pass "
                "--model_code_roots MonoViT=/path/to/MonoViT."
            ) from exc

        self.torch = torch
        self.disp_to_depth = disp_to_depth
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        encoder_path = weights_folder / "encoder.pth"
        decoder_path = weights_folder / "depth.pth"
        if not encoder_path.is_file() or not decoder_path.is_file():
            raise FileNotFoundError(
                f"{name}: expected encoder.pth and depth.pth inside {weights_folder}"
            )

        if not hasattr(networks, "mpvit_small"):
            raise RuntimeError(f"{name}: networks.mpvit_small is not available in {code_root}")

        encoder_dict = self._unwrap_state_dict(self._load_torch_file(encoder_path))
        decoder_dict = self._unwrap_state_dict(self._load_torch_file(decoder_path))
        self.encoder = networks.mpvit_small()
        self.encoder.num_ch_enc = [64, 128, 216, 288, 288]
        if isinstance(encoder_dict, dict):
            self.height = int(encoder_dict.get("height", self.height))
            self.width = int(encoder_dict.get("width", self.width))
        self.depth_decoder = self._build_best_decoder(networks, decoder_dict, name)
        self._load_partial(self.encoder, encoder_dict, model_name=f"{name}/encoder")

        self.encoder.to(self.device).eval()
        self.depth_decoder.to(self.device).eval()

    def _decoder_constructors(self, networks):
        classes = []
        modules = [networks]
        if hasattr(networks, "__path__"):
            for module_info in pkgutil.iter_modules(networks.__path__):
                try:
                    modules.append(importlib.import_module(f"{networks.__name__}.{module_info.name}"))
                except Exception:
                    continue

        seen = set()
        for module in modules:
            for attr_name in dir(module):
                if "decoder" not in attr_name.lower():
                    continue
                obj = getattr(module, attr_name)
                if not inspect.isclass(obj):
                    continue
                try:
                    if not issubclass(obj, self.torch.nn.Module):
                        continue
                except TypeError:
                    continue
                key = f"{obj.__module__}.{obj.__name__}"
                if key in seen:
                    continue
                seen.add(key)
                classes.append((obj.__name__, obj))

        classes.sort(key=lambda item: (item[0] != "DepthDecoder", item[0]))

        attempts = []
        for class_name, cls in classes:
            attempts.extend(
                [
                    (class_name, lambda cls=cls: cls(self.encoder.num_ch_enc, scales=range(4))),
                    (class_name, lambda cls=cls: cls(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))),
                    (class_name, lambda cls=cls: cls(self.encoder.num_ch_enc)),
                    (class_name, lambda cls=cls: cls(num_ch_enc=self.encoder.num_ch_enc)),
                    (class_name, lambda cls=cls: cls(scales=range(4))),
                    (class_name, lambda cls=cls: cls()),
                ]
            )
        return attempts

    def _build_best_decoder(self, networks, decoder_dict, name: str):
        best = None
        failures = []
        for class_name, ctor in self._decoder_constructors(networks):
            try:
                decoder = ctor()
                filtered, audit = self._match_state_dict(
                    decoder,
                    decoder_dict,
                    model_name=f"{name}/depth[{class_name}]",
                )
            except Exception as exc:
                if len(failures) < 8:
                    failures.append(f"{class_name}: {exc}")
                continue

            score = (
                audit["loaded_keys"],
                -audit["missing_model_keys"],
                -audit["shape_mismatch_count"],
                -audit["unexpected_checkpoint_keys"],
            )
            if best is None or score > best[0]:
                best = (score, class_name, decoder, filtered, audit)

        if best is None:
            raise RuntimeError(
                f"{name}: no decoder class matched depth.pth. Tried: {failures}"
            )

        _, class_name, decoder, filtered, audit = best
        decoder.load_state_dict(filtered, strict=False)
        audit["module"] = f"{name}/depth"
        audit["selected_decoder_class"] = class_name
        self.load_audit.append(audit)
        print(
            f"[INFO] {name}: selected decoder {class_name} "
            f"({audit['loaded_keys']}/{audit['model_keys']} keys loaded)"
        )
        return decoder


class EndoDacPredictor(Monodepth2Predictor):
    def __init__(
        self,
        name: str,
        weights_folder: Path,
        code_root: Optional[Path],
        height: int,
        width: int,
        min_depth: float,
        max_depth: float,
        device: str,
        output_mode: str,
        pretrained_path: Optional[Path],
    ) -> None:
        self.name = name
        self.weights_folder = weights_folder
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device_name = device
        self.output_mode = output_mode
        self.load_audit = []

        prepare_import_path(code_root, purge_prefixes=["models", "utils"])

        try:
            import torch
            import models.endodac as endodac
            from utils.layers import disp_to_depth
        except Exception as exc:  # pragma: no cover - depends on the VM repo.
            raise RuntimeError(
                "Could not import ENDO-DAC modules. Pass "
                "--model_code_roots ENDO-DAC=/workspace/ENDO-DAC."
            ) from exc

        self.torch = torch
        self.disp_to_depth = disp_to_depth
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        depther_path = weights_folder / "depth_model.pth"
        if not depther_path.is_file():
            raise FileNotFoundError(f"{name}: expected depth_model.pth inside {weights_folder}")
        depther_dict = self._unwrap_state_dict(self._load_torch_file(depther_path))
        if isinstance(depther_dict, dict):
            self.height = int(depther_dict.get("height", self.height))
            self.width = int(depther_dict.get("width", self.width))

        if pretrained_path is None and code_root is not None:
            pretrained_path = code_root / "pretrained_model"
        if pretrained_path is None:
            pretrained_path = weights_folder

        self.depther = endodac.endodac(
            backbone_size="base",
            r=4,
            lora_type="dvlora",
            image_shape=(224, 280),
            pretrained_path=str(pretrained_path),
            residual_block_indexes=[2, 5, 8, 11],
            include_cls_token=True,
        )
        self._load_partial(self.depther, depther_dict, model_name=f"{name}/depth_model")
        self.depther.to(self.device).eval()

    def predict(self, rgb: Image.Image) -> np.ndarray:
        image = rgb.convert("RGB").resize((self.width, self.height), Image.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            output = self.depther(tensor)
            disp = output[("disp", 0)]
            scaled_disp, depth = self.disp_to_depth(disp, self.min_depth, self.max_depth)
            pred = depth[0, 0].detach().cpu().numpy() if self.output_mode == "depth" else scaled_disp[0, 0].detach().cpu().numpy()

        pred_img = Image.fromarray(pred.astype(np.float32), mode="F")
        pred_img = pred_img.resize(rgb.size, Image.BILINEAR)
        return np.asarray(pred_img).astype(np.float32)


class ManyDepthPredictor(Monodepth2Predictor):
    def __init__(
        self,
        name: str,
        weights_folder: Path,
        code_root: Optional[Path],
        height: int,
        width: int,
        min_depth: float,
        max_depth: float,
        device: str,
        output_mode: str,
        mode: str,
    ) -> None:
        self.name = name
        self.weights_folder = weights_folder
        self.height = height
        self.width = width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device_name = device
        self.output_mode = output_mode
        self.mode = mode
        self.load_audit = []

        if code_root is not None and code_root.name.lower() == "manydepth":
            code_root = code_root.parent
        prepare_import_path(code_root, purge_prefixes=["manydepth"])

        try:
            import torch
            from manydepth import networks
            from manydepth.layers import disp_to_depth
        except Exception as exc:  # pragma: no cover - depends on the VM repo.
            raise RuntimeError(
                "Could not import manydepth. Pass --model_code_roots "
                "ManyDepth=/workspace/endo-manydepth/endo-manydepth-master."
            ) from exc

        self.torch = torch
        self.disp_to_depth = disp_to_depth
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        encoder_path = weights_folder / "encoder.pth"
        decoder_path = weights_folder / "depth.pth"
        if not encoder_path.is_file() or not decoder_path.is_file():
            raise FileNotFoundError(
                f"{name}: expected encoder.pth and depth.pth inside {weights_folder}"
            )

        encoder_dict = self._unwrap_state_dict(self._load_torch_file(encoder_path))
        self.feed_height = int(encoder_dict.get("height", height))
        self.feed_width = int(encoder_dict.get("width", width))
        min_bin = float(encoder_dict.get("min_depth_bin", 0.1))
        max_bin = float(encoder_dict.get("max_depth_bin", 20.0))

        self.encoder = networks.ResnetEncoderMatching(
            18,
            False,
            input_width=self.feed_width,
            input_height=self.feed_height,
            adaptive_bins=True,
            min_depth_bin=min_bin,
            max_depth_bin=max_bin,
            depth_binning="linear",
            num_depth_bins=96,
        )
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        decoder_dict = self._unwrap_state_dict(self._load_torch_file(decoder_path))
        self._load_partial(self.encoder, encoder_dict, model_name=f"{name}/encoder")
        self._load_partial(self.depth_decoder, decoder_dict, model_name=f"{name}/depth")

        self.encoder.to(self.device).eval()
        self.depth_decoder.to(self.device).eval()
        self.min_depth_bin = min_bin
        self.max_depth_bin = max_bin

    def predict(self, rgb: Image.Image) -> np.ndarray:
        image = rgb.convert("RGB").resize((self.feed_width, self.feed_height), Image.LANCZOS)
        arr = np.asarray(image).astype(np.float32) / 255.0
        input_image = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        source = self.torch.zeros_like(input_image).unsqueeze(1)
        pose = self.torch.zeros((1, 1, 4, 4), dtype=input_image.dtype, device=self.device)
        pose[:, :, 0, 0] = 1.0
        pose[:, :, 1, 1] = 1.0
        pose[:, :, 2, 2] = 1.0
        pose[:, :, 3, 3] = 1.0

        k_np = np.eye(4, dtype=np.float32)
        k_np[0, 0] = self.feed_width / 4.0
        k_np[1, 1] = self.feed_height / 4.0
        k_np[0, 2] = self.feed_width / 8.0
        k_np[1, 2] = self.feed_height / 8.0
        k = self.torch.from_numpy(k_np).unsqueeze(0).to(self.device)
        inv_k = self.torch.inverse(k)

        with self.torch.no_grad():
            output, _, _ = self.encoder(
                current_image=input_image,
                lookup_images=source,
                poses=pose,
                K=k,
                invK=inv_k,
                min_depth_bin=self.min_depth_bin,
                max_depth_bin=self.max_depth_bin,
            )
            output = self.depth_decoder(output)
            disp = output[("disp", 0)]
            scaled_disp, depth = self.disp_to_depth(disp, self.min_depth, self.max_depth)
            pred = depth[0, 0].detach().cpu().numpy() if self.output_mode == "depth" else scaled_disp[0, 0].detach().cpu().numpy()

        pred_img = Image.fromarray(pred.astype(np.float32), mode="F")
        pred_img = pred_img.resize(rgb.size, Image.BILINEAR)
        return np.asarray(pred_img).astype(np.float32)


class EndoSfmLearnerPredictor(Monodepth2Predictor):
    def __init__(
        self,
        name: str,
        weights_folder: Path,
        code_root: Optional[Path],
        height: int,
        width: int,
        device: str,
    ) -> None:
        self.name = name
        self.weights_folder = weights_folder
        self.height = height
        self.width = width
        self.load_audit = []

        prepare_import_path(code_root, purge_prefixes=["models"])

        try:
            import torch
            import models
        except Exception as exc:  # pragma: no cover - depends on the VM repo.
            raise RuntimeError(
                "Could not import EndoSfMLearner models. Pass "
                "--model_code_roots EndoSfMLearner=/path/to/EndoSfMLearner."
            ) from exc

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        checkpoint_path = self._find_dispnet_checkpoint(weights_folder)
        checkpoint = self._load_torch_file(checkpoint_path)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise RuntimeError(f"{name}: {checkpoint_path} does not contain a state_dict")

        self.disp_net = models.DispResNet(18, False).to(self.device)
        result = self.disp_net.load_state_dict(checkpoint["state_dict"])
        self.load_audit.append(
            {
                "module": f"{name}/dispnet",
                "checkpoint_keys": len(checkpoint["state_dict"]),
                "model_keys": len(self.disp_net.state_dict()),
                "loaded_keys": len(checkpoint["state_dict"]),
                "missing_model_keys": len(list(getattr(result, "missing_keys", []))),
                "unexpected_checkpoint_keys": len(list(getattr(result, "unexpected_keys", []))),
                "shape_mismatch_count": 0,
                "missing_model_keys_sample": list(getattr(result, "missing_keys", []))[:12],
                "unexpected_checkpoint_keys_sample": list(getattr(result, "unexpected_keys", []))[:12],
                "shape_mismatches_sample": [],
            }
        )
        self.disp_net.eval()

    @staticmethod
    def _find_dispnet_checkpoint(weights_folder: Path) -> Path:
        candidates = [
            weights_folder / "dispnet_model_best.pth.tar",
            weights_folder / "dispnet_checkpoint.pth.tar",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for current_root, _, files in os.walk(weights_folder):
            for name in ("dispnet_model_best.pth.tar", "dispnet_checkpoint.pth.tar"):
                if name in files:
                    return Path(current_root) / name
        raise FileNotFoundError(
            f"expected dispnet_model_best.pth.tar or dispnet_checkpoint.pth.tar inside {weights_folder}"
        )

    def predict(self, rgb: Image.Image) -> np.ndarray:
        image = rgb.convert("RGB").resize((self.width, self.height), Image.LANCZOS)
        arr = np.asarray(image).astype(np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        tensor = ((self.torch.from_numpy(arr).unsqueeze(0) / 255.0 - 0.45) / 0.225).to(self.device)

        with self.torch.no_grad():
            pred = self.disp_net(tensor).detach().cpu().numpy()[0, 0].astype(np.float32)

        pred_img = Image.fromarray(pred, mode="F")
        pred_img = pred_img.resize(rgb.size, Image.BILINEAR)
        return np.asarray(pred_img).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a qualitative corruption grid: Input column plus one column per "
            "model/prediction source."
        )
    )
    parser.add_argument("--corruptions_root", required=True)
    parser.add_argument("--output", default="qualitative_corruption_grid.png")

    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        metavar="NAME=WEIGHTS_FOLDER",
        help="Monodepth2-style weights folders containing encoder.pth and depth.pth.",
    )
    parser.add_argument(
        "--weights_backup_root",
        default="/workspace/weight",
        help=(
            "Fallback directory with weight folders or .zip backups. Used only "
            "when a --models path is missing or does not contain encoder.pth/depth.pth."
        ),
    )
    parser.add_argument(
        "--weights_extract_root",
        default=None,
        help="Where backup .zip files are extracted. Defaults to BACKUP_ROOT/_extracted.",
    )
    parser.add_argument(
        "--prediction_roots",
        nargs="*",
        default=[],
        metavar="NAME=PRED_ROOT",
        help=(
            "Precomputed prediction roots. Files are matched by corruption, severity "
            "and relative image path; accepts npy/npz/png/jpg/tif."
        ),
    )
    parser.add_argument(
        "--models_json",
        default=None,
        help=(
            "Optional JSON list with entries like "
            "[{\"name\":\"MonoDepth2\",\"kind\":\"monodepth2\",\"path\":\"/w\"}]."
        ),
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=[],
        metavar="NAME=TYPE",
        help="Optional per-model type: monodepth2, monovit, endodac, manydepth, endosfm.",
    )
    parser.add_argument(
        "--model_code_roots",
        nargs="*",
        default=[],
        metavar="NAME=CODE_ROOT",
        help="Optional per-model code root, e.g. ENDO-DAC=/workspace/ENDO-DAC.",
    )
    parser.add_argument(
        "--endodac_pretrained_paths",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Optional ENDO-DAC pretrained_model folder per model.",
    )

    parser.add_argument(
        "--code_root",
        default=None,
        help="Repo containing networks.py/layers.py for monodepth2-style models.",
    )
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_output",
        choices=["disp", "depth"],
        default="disp",
        help="What to visualize from monodepth2-style models.",
    )
    parser.add_argument(
        "--manydepth_mode",
        choices=["mono"],
        default="mono",
        help="ManyDepth qualitative mode. Mono uses a zero cost volume for single-frame figures.",
    )
    parser.add_argument(
        "--print_load_audit",
        action="store_true",
        help="Print loaded/missing/mismatched key counts for every model component.",
    )
    parser.add_argument(
        "--print_prediction_stats",
        action="store_true",
        help="Print numeric prediction stats for the first rows.",
    )
    parser.add_argument(
        "--prediction_stats_rows",
        type=int,
        default=1,
        help="How many corruption rows to print when --print_prediction_stats is set.",
    )

    parser.add_argument("--split_file", default=None)
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument(
        "--rel_image",
        default=None,
        help="Relative image path under each severity data root. Overrides split_file.",
    )
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png")
    parser.add_argument("--nested_data_dir", default="endovis_data")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument(
        "--corruptions",
        default="all",
        help="Comma-separated corruption dirs or 'all'.",
    )

    parser.add_argument("--cell_width", type=int, default=180)
    parser.add_argument("--cell_height", type=int, default=88)
    parser.add_argument("--header_height", type=int, default=26)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--font_size", type=int, default=14)
    parser.add_argument("--label_font_size", type=int, default=13)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--normalize_low", type=float, default=2.0)
    parser.add_argument("--normalize_high", type=float, default=98.0)
    parser.add_argument(
        "--invert_prediction_files",
        action="store_true",
        help="Invert loaded prediction files before colorizing, useful for depth maps.",
    )
    parser.add_argument(
        "--missing_policy",
        choices=["placeholder", "error"],
        default="placeholder",
    )
    parser.add_argument("--caption", default=None)
    parser.add_argument("--save_metadata", action="store_true")
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="If set, log the generated grid to this Weights & Biases project.",
    )
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument(
        "--wandb_key",
        default="qualitative_corruption_grid",
        help="Metric/media key used when logging the grid image.",
    )
    return parser.parse_args()


def parse_key_value_specs(values: Iterable[str], kind: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        specs.append(ModelSpec(name=name, kind=kind, path=Path(path).expanduser()))
    return specs


def parse_name_map(values: Iterable[str]) -> dict[str, str]:
    mapping = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        name = normalize_match_text(name)
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        mapping[name] = value
    return mapping


def infer_model_kind(spec: ModelSpec, explicit_types: dict[str, str]) -> str:
    key = normalize_match_text(spec.name)
    if key in explicit_types:
        return explicit_types[key].strip().lower()

    joined = normalize_match_text(f"{spec.name} {spec.path}")
    if "endodac" in joined:
        return "endodac"
    if "endosfm" in joined or "endosfmlearner" in joined:
        return "endosfm"
    if "monovit" in joined:
        return "monovit"
    if "manydepth" in joined:
        return "manydepth"
    return spec.kind


def load_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = []
    specs.extend(parse_key_value_specs(args.models, "monodepth2"))
    specs.extend(parse_key_value_specs(args.prediction_roots, "predictions"))

    if args.models_json:
        with open(args.models_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError("--models_json must be a JSON list")
        for item in loaded:
            specs.append(
                ModelSpec(
                    name=str(item["name"]),
                    kind=str(item.get("kind", "monodepth2")),
                    path=Path(str(item["path"])).expanduser(),
                )
            )

    if not specs:
        raise ValueError("Add at least one --models or --prediction_roots entry.")

    explicit_types = parse_name_map(args.model_types)
    for spec in specs:
        if spec.kind != "predictions":
            spec.kind = infer_model_kind(spec, explicit_types)
    return specs


def natural_corruption_key(name: str) -> tuple[int, str]:
    low = name.lower()
    if low in CORRUPTION_ORDER:
        return (CORRUPTION_ORDER.index(low), low)
    return (len(CORRUPTION_ORDER), low)


def existing_corruptions(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"corruptions_root not found: {root}")
    names = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(names, key=natural_corruption_key)


def select_corruptions(root: Path, requested: str) -> list[str]:
    available = existing_corruptions(root)
    if requested.strip().lower() == "all":
        return available

    wanted = [x.strip() for x in requested.split(",") if x.strip()]
    lower_to_actual = {x.lower(): x for x in available}
    selected = []
    missing = []
    for item in wanted:
        actual = lower_to_actual.get(item.lower())
        if actual is None:
            missing.append(item)
        else:
            selected.append(actual)

    if missing:
        raise FileNotFoundError(f"Corruption dirs not found: {missing}. Available: {available}")
    return selected


def severity_root(corruptions_root: Path, corruption: str, severity: int) -> Path:
    return corruptions_root / corruption / f"severity_{severity}"


def data_root_for_severity(sev_root: Path, nested_data_dir: str) -> Path:
    candidates = []
    if nested_data_dir:
        candidates.append(sev_root / nested_data_dir)
    candidates.extend([sev_root / "endovis_data", sev_root])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def normalize_rel_path(path: str) -> str:
    path = path.strip().replace("\\", "/").strip("/")
    return path


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def extension_list(raw: str) -> tuple[str, ...]:
    exts = []
    for ext in raw.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext)
    return tuple(exts) or IMAGE_EXTENSIONS


def frame_stems(token: str) -> list[str]:
    token = os.path.splitext(str(token).strip())[0]
    stems = [token]
    try:
        value = int(token)
        stems.extend([str(value), f"{value:04d}", f"{value:06d}", f"{value:010d}"])
    except ValueError:
        pass
    out = []
    for stem in stems:
        if stem not in out:
            out.append(stem)
    return out


def resolve_explicit_rel(root: Path, rel_image: str) -> Optional[tuple[Path, str]]:
    rel = normalize_rel_path(rel_image)
    candidates = [root / rel]
    parts = rel.split("/")
    if len(parts) > 1 and parts[0] == "endovis_data":
        candidates.append(root / "/".join(parts[1:]))
    for candidate in unique_paths(candidates):
        if candidate.is_file():
            return candidate, normalize_rel_path(str(candidate.relative_to(root)))
    return None


def resolve_split_line(root: Path, line: str, exts: tuple[str, ...]) -> Optional[tuple[Path, str]]:
    parts = line.strip().split()
    if not parts:
        return None

    first = normalize_rel_path(parts[0])
    candidates: list[Path] = []

    if Path(first).suffix.lower() in IMAGE_EXTENSIONS:
        candidates.append(root / first)

    if len(parts) >= 2:
        folder = first
        frame = parts[1]
        dirs = [
            folder,
            f"{folder}/data",
            f"{folder}/image01",
            f"{folder}/image02",
        ]
        if folder.endswith("/data") or folder.endswith("/image01") or folder.endswith("/image02"):
            dirs.insert(0, folder)
        for directory in dirs:
            for stem in frame_stems(frame):
                if Path(frame).suffix.lower() in IMAGE_EXTENSIONS:
                    candidates.append(root / directory / frame)
                for ext in exts:
                    candidates.append(root / directory / f"{stem}{ext}")

    for candidate in unique_paths(candidates):
        if candidate.is_file():
            return candidate, normalize_rel_path(str(candidate.relative_to(root)))
    return None


def first_image_under(root: Path, exts: tuple[str, ...]) -> Optional[tuple[Path, str]]:
    for current_root, _, files in os.walk(root):
        current = Path(current_root)
        for name in sorted(files):
            if Path(name).suffix.lower() in exts:
                path = current / name
                return path, normalize_rel_path(str(path.relative_to(root)))
    return None


def pick_reference_image(
    root: Path,
    rel_image: Optional[str],
    split_file: Optional[str],
    split_index: int,
    exts: tuple[str, ...],
) -> tuple[Path, str]:
    if rel_image:
        found = resolve_explicit_rel(root, rel_image)
        if found:
            return found
        raise FileNotFoundError(f"Could not resolve --rel_image {rel_image} under {root}")

    if split_file:
        with open(split_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            raise RuntimeError(f"split file is empty: {split_file}")
        if split_index < 0 or split_index >= len(lines):
            raise IndexError(f"--split_index {split_index} outside split length {len(lines)}")
        found = resolve_split_line(root, lines[split_index], exts)
        if found:
            return found
        raise FileNotFoundError(
            f"Could not resolve split line {split_index}: {lines[split_index]} under {root}"
        )

    found = first_image_under(root, exts)
    if found:
        return found
    raise FileNotFoundError(f"No image found under {root}")


def resolve_row_image(
    data_root: Path,
    rel_path: str,
    rel_image: Optional[str],
    split_file: Optional[str],
    split_index: int,
    exts: tuple[str, ...],
) -> tuple[Path, str]:
    candidate = data_root / rel_path
    if candidate.is_file():
        return candidate, rel_path
    if rel_image or split_file:
        return pick_reference_image(data_root, rel_image, split_file, split_index, exts)
    found = first_image_under(data_root, exts)
    if found:
        return found
    raise FileNotFoundError(f"No row image found under {data_root}")


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_to_cell(img: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    cell_w, cell_h = cell_size
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(cell_w / src_w, cell_h / src_h)
    new_size = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))
    resized = img.resize(new_size, Image.LANCZOS)
    left = max(0, (resized.size[0] - cell_w) // 2)
    top = max(0, (resized.size[1] - cell_h) // 2)
    return resized.crop((left, top, left + cell_w, top + cell_h))


def safe_percentiles(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_map(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = values.astype(np.float32)
    lo, hi = safe_percentiles(values, low, high)
    norm = (np.clip(values, lo, hi) - lo) / (hi - lo + 1e-8)
    norm[~np.isfinite(norm)] = 0.0
    return norm


def colormap_array(norm: np.ndarray, cmap_name: str) -> np.ndarray:
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgb = (cmap(np.clip(norm, 0.0, 1.0))[..., :3] * 255.0).astype(np.uint8)
        return rgb
    except Exception:
        # Lightweight fallback: black -> purple -> orange -> yellow.
        x = np.clip(norm[..., None], 0.0, 1.0)
        stops = np.array(
            [
                [0, 0, 4],
                [80, 18, 123],
                [182, 54, 121],
                [251, 136, 97],
                [252, 253, 191],
            ],
            dtype=np.float32,
        )
        pos = np.clip(x * (len(stops) - 1), 0, len(stops) - 1 - 1e-6)
        idx = np.floor(pos).astype(np.int32)
        frac = pos - idx
        rgb = stops[idx[..., 0]] * (1.0 - frac) + stops[idx[..., 0] + 1] * frac
        return rgb.astype(np.uint8)


def prediction_to_image(
    values: np.ndarray,
    cell_size: tuple[int, int],
    cmap: str,
    low: float,
    high: float,
    invert: bool,
) -> Image.Image:
    values = values.astype(np.float32)
    if invert:
        finite = np.isfinite(values)
        out = np.zeros_like(values, dtype=np.float32)
        out[finite] = 1.0 / np.maximum(values[finite], 1e-8)
        values = out
    norm = normalize_map(values, low, high)
    rgb = colormap_array(norm, cmap)
    return resize_to_cell(Image.fromarray(rgb, mode="RGB"), cell_size)


def load_prediction_file(path: Path) -> np.ndarray | Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return np.squeeze(arr)
    if suffix == ".npz":
        data = np.load(path)
        key = "data" if "data" in data else data.files[0]
        return np.squeeze(data[key])

    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return img.convert("RGB")
    return arr.astype(np.float32)


def prediction_candidates(
    pred_root: Path,
    corruption: str,
    severity: int,
    rel_path: str,
) -> list[Path]:
    rel = Path(rel_path)
    severity_dir = f"severity_{severity}"
    bases = [
        pred_root / corruption / severity_dir / rel,
        pred_root / corruption / severity_dir / "endovis_data" / rel,
        pred_root / severity_dir / rel,
        pred_root / severity_dir / "endovis_data" / rel,
        pred_root / corruption / rel,
        pred_root / rel,
    ]

    candidates = []
    for base in bases:
        candidates.append(base)
        for ext in DEPTH_EXTENSIONS:
            candidates.append(base.with_suffix(ext))
    return unique_paths(candidates)


def find_prediction(pred_root: Path, corruption: str, severity: int, rel_path: str) -> Optional[Path]:
    for path in prediction_candidates(pred_root, corruption, severity, rel_path):
        if path.is_file():
            return path
    return None


def placeholder_cell(
    text: str,
    cell_size: tuple[int, int],
    font: ImageFont.ImageFont,
    bg=(245, 245, 245),
) -> Image.Image:
    img = Image.new("RGB", cell_size, bg)
    draw = ImageDraw.Draw(img)
    lines = []
    current = ""
    for word in text.split():
        tentative = word if not current else current + " " + word
        if draw.textbbox((0, 0), tentative, font=font)[2] <= cell_size[0] - 12:
            current = tentative
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    total_h = len(lines) * (font.size + 3 if hasattr(font, "size") else 14)
    y = max(6, (cell_size[1] - total_h) // 2)
    for line in lines[:4]:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (cell_size[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=(90, 90, 90), font=font)
        y += (bbox[3] - bbox[1]) + 5
    return img


def draw_header(
    canvas: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=3, fill=fill)
    bbox = draw.textbbox((0, 0), label, font=font)
    tx = x + (width - (bbox[2] - bbox[0])) // 2
    ty = y + (height - (bbox[3] - bbox[1])) // 2 - 1
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)


def draw_row_label(cell: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    cell = cell.copy()
    draw = ImageDraw.Draw(cell, "RGBA")
    bbox = draw.textbbox((0, 0), label, font=font)
    pad_x = 6
    pad_y = 3
    rect = (0, 0, bbox[2] - bbox[0] + pad_x * 2, bbox[3] - bbox[1] + pad_y * 2)
    draw.rectangle(rect, fill=(67, 92, 204, 205))
    draw.text((pad_x, pad_y - 1), label, fill=(255, 255, 255, 255), font=font)
    return cell.convert("RGB")


def display_label(corruption: str) -> str:
    return DISPLAY_LABELS.get(corruption.lower(), corruption.replace("_", " ").title())


def valid_weights_folder(path: Path, kind: str = "monodepth2") -> bool:
    if not path.is_dir():
        return False
    if kind == "endodac":
        return (path / "depth_model.pth").is_file()
    if kind == "endosfm":
        return (
            (path / "dispnet_model_best.pth.tar").is_file()
            or (path / "dispnet_checkpoint.pth.tar").is_file()
        )
    return (path / "encoder.pth").is_file() and (path / "depth.pth").is_file()


def normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def model_aliases(name: str) -> list[str]:
    key = normalize_match_text(name)
    aliases = [key]
    known_aliases = {
        "afsfmlearner": ["afmlearner", "afsfm", "sfmlearner"],
        "afmlearner": ["afsfmlearner", "afsfm", "sfmlearner"],
        "endosfmlearner": ["endosfmlearner", "scaredendo", "endosfm"],
        "monodepth2": ["monodepth2", "m2"],
        "monovit": ["monovit", "weights19monovit"],
        "endodac": ["endodac", "dac"],
        "manydepth": ["manydepth"],
        "owners": ["owners", "monoiit", "own"],
    }
    for canonical, values in known_aliases.items():
        if canonical in key or key in canonical or any(v in key for v in values):
            aliases.extend([canonical, *values])

    out = []
    for alias in aliases:
        alias = normalize_match_text(alias)
        if alias and alias not in out:
            out.append(alias)
    return out


def candidate_score(candidate: Path, spec: ModelSpec) -> int:
    haystack = normalize_match_text(" ".join(candidate.parts[-4:]))
    aliases = model_aliases(spec.name)
    path_hint = normalize_match_text(spec.path.name)
    score = 0
    for alias in aliases:
        if alias and alias in haystack:
            score += 100 + len(alias)
        elif alias and haystack in alias:
            score += 20
    if path_hint and path_hint in haystack:
        score += 40
    if candidate.suffix.lower() == ".zip":
        score += 5
    return score


def find_valid_weight_dirs(root: Path, kind: str = "monodepth2") -> list[Path]:
    if not root.is_dir():
        return []
    found = []
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        if valid_weights_folder(current, kind):
            found.append(current)
            dirs[:] = []
    return found


def pick_best_candidate(candidates: list[Path], spec: ModelSpec) -> Optional[Path]:
    best = pick_best_scored_candidate(candidates, spec)
    return best[1] if best is not None else None


def pick_best_scored_candidate(
    candidates: list[Path],
    spec: ModelSpec,
) -> Optional[tuple[int, Path]]:
    scored = [(candidate_score(path, spec), path) for path in candidates]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(str(item[1]))), reverse=True)
    return scored[0]


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path inside zip: {member.filename}")
        zf.extractall(extract_dir)


def resolve_backup_weights_folder(spec: ModelSpec, args: argparse.Namespace) -> Path:
    original = spec.path.expanduser()
    if valid_weights_folder(original, spec.kind):
        return original

    if original.is_dir():
        nested = find_valid_weight_dirs(original, spec.kind)
        best_nested = pick_best_candidate(nested, spec) or (nested[0] if nested else None)
        if best_nested is not None:
            print(f"[INFO] {spec.name}: using nested weights folder {best_nested}")
            return best_nested

    backup_root = Path(args.weights_backup_root).expanduser() if args.weights_backup_root else None
    if backup_root is None or not backup_root.is_dir():
        return original

    best_dir = pick_best_scored_candidate(find_valid_weight_dirs(backup_root, spec.kind), spec)
    zip_candidates = [p for p in backup_root.glob("*.zip") if p.is_file()]
    best_zip = pick_best_scored_candidate(zip_candidates, spec)

    if best_dir is not None and (best_zip is None or best_dir[0] >= best_zip[0]):
        print(f"[INFO] {spec.name}: path not found, using backup folder {best_dir[1]}")
        return best_dir[1]

    if best_zip is None:
        return original

    extract_root = (
        Path(args.weights_extract_root).expanduser()
        if args.weights_extract_root
        else backup_root / "_extracted"
    )
    extract_dir = extract_root / best_zip[1].stem
    valid_inside = find_valid_weight_dirs(extract_dir, spec.kind)
    if not valid_inside:
        print(f"[INFO] {spec.name}: extracting backup {best_zip[1]} -> {extract_dir}")
        safe_extract_zip(best_zip[1], extract_dir)
        valid_inside = find_valid_weight_dirs(extract_dir, spec.kind)

    best_inside = pick_best_candidate(valid_inside, spec) or (valid_inside[0] if valid_inside else None)
    if best_inside is not None:
        print(f"[INFO] {spec.name}: using extracted backup weights {best_inside}")
        return best_inside

    return original


def build_predictors(args: argparse.Namespace, specs: list[ModelSpec]):
    predictors = {}
    global_code_root = Path(args.code_root).expanduser() if args.code_root else None
    code_root_map = {
        key: Path(value).expanduser()
        for key, value in parse_name_map(args.model_code_roots).items()
    }
    endodac_pretrained_map = {
        key: Path(value).expanduser()
        for key, value in parse_name_map(args.endodac_pretrained_paths).items()
    }

    for spec in specs:
        model_key = normalize_match_text(spec.name)
        code_root = code_root_map.get(model_key, global_code_root)
        if code_root is None and spec.kind == "endodac":
            for parent in [spec.path, *spec.path.parents]:
                if parent.name == "ENDO-DAC":
                    code_root = parent
                    break
        if code_root is None and spec.kind == "manydepth":
            for parent in [spec.path, *spec.path.parents]:
                if parent.name.lower() == "manydepth":
                    code_root = parent.parent
                    break

        if code_root is None and spec.kind == "endosfm":
            for parent in [spec.path, *spec.path.parents]:
                if parent.name == "EndoSfMLearner":
                    code_root = parent
                    break
        spec.code_root = code_root

        if spec.kind in {"monodepth2", "monovit", "endodac", "manydepth", "endosfm"}:
            spec.path = resolve_backup_weights_folder(spec, args)
            spec.pretrained_path = endodac_pretrained_map.get(model_key)
            spec.checkpoint_files = model_checkpoint_files(spec)
            try:
                if spec.kind == "monovit":
                    predictors[spec.name] = MonoViTPredictor(
                        name=spec.name,
                        weights_folder=spec.path,
                        code_root=code_root,
                        height=args.height,
                        width=args.width,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        device=args.device,
                        output_mode=args.model_output,
                    )
                elif spec.kind == "endodac":
                    predictors[spec.name] = EndoDacPredictor(
                        name=spec.name,
                        weights_folder=spec.path,
                        code_root=code_root,
                        height=args.height,
                        width=args.width,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        device=args.device,
                        output_mode=args.model_output,
                        pretrained_path=spec.pretrained_path,
                    )
                elif spec.kind == "manydepth":
                    predictors[spec.name] = ManyDepthPredictor(
                        name=spec.name,
                        weights_folder=spec.path,
                        code_root=code_root,
                        height=args.height,
                        width=args.width,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        device=args.device,
                        output_mode=args.model_output,
                        mode=args.manydepth_mode,
                    )
                elif spec.kind == "endosfm":
                    predictors[spec.name] = EndoSfmLearnerPredictor(
                        name=spec.name,
                        weights_folder=spec.path,
                        code_root=code_root,
                        height=args.height,
                        width=args.width,
                        device=args.device,
                    )
                else:
                    predictors[spec.name] = Monodepth2Predictor(
                        name=spec.name,
                        weights_folder=spec.path,
                        code_root=code_root,
                        num_layers=args.num_layers,
                        height=args.height,
                        width=args.width,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        device=args.device,
                        output_mode=args.model_output,
                    )
                predictor = predictors[spec.name]
                spec.load_audit = list(getattr(predictor, "load_audit", []))
            except Exception as exc:
                if args.missing_policy == "error":
                    raise
                predictors[spec.name] = exc
                print(f"[WARN] {spec.name}: could not initialize model, cells will be placeholders: {exc}")
                spec.load_audit = [{"error": str(exc)}]
        elif spec.kind == "predictions":
            predictors[spec.name] = None
        else:
            raise ValueError(f"Unsupported model kind: {spec.kind}")
    return predictors


def model_checkpoint_files(spec: ModelSpec) -> list[str]:
    if spec.kind in {"monodepth2", "monovit", "manydepth"}:
        return [
            str(spec.path / "encoder.pth"),
            str(spec.path / "depth.pth"),
        ]
    if spec.kind == "endodac":
        files = [str(spec.path / "depth_model.pth")]
        pretrained = spec.pretrained_path or spec.path
        files.append(str(pretrained / "depth_anything_vitb14.pth"))
        return files
    if spec.kind == "endosfm":
        try:
            return [str(EndoSfmLearnerPredictor._find_dispnet_checkpoint(spec.path))]
        except Exception:
            return [
                str(spec.path / "dispnet_model_best.pth.tar"),
                str(spec.path / "dispnet_checkpoint.pth.tar"),
            ]
    return []


def print_model_summary(specs: list[ModelSpec]) -> None:
    print("\n======= MODEL SUMMARY =======")
    for spec in specs:
        print(f"- {spec.name}")
        print(f"  kind: {spec.kind}")
        print(f"  weights_path: {spec.path}")
        if spec.code_root is not None:
            print(f"  code_root: {spec.code_root}")
        if spec.pretrained_path is not None:
            print(f"  pretrained_path: {spec.pretrained_path}")
        if spec.checkpoint_files:
            print("  checkpoint_files:")
            for path in spec.checkpoint_files:
                print(f"    - {path}")


def print_load_audit(specs: list[ModelSpec]) -> None:
    print("\n======= LOAD AUDIT =======")
    for spec in specs:
        print(f"- {spec.name}")
        if not spec.load_audit:
            print("  no load audit recorded")
            continue
        for audit in spec.load_audit:
            if "error" in audit:
                print(f"  ERROR: {audit['error']}")
                continue
            print(
                "  {module}: loaded {loaded_keys}/{model_keys} model keys "
                "from {checkpoint_keys} checkpoint keys | missing={missing_model_keys} "
                "| unexpected={unexpected_checkpoint_keys} | shape_mismatch={shape_mismatch_count}".format(
                    **audit
                )
            )
            if audit["missing_model_keys_sample"]:
                print(f"    missing sample: {audit['missing_model_keys_sample']}")
            if audit["unexpected_checkpoint_keys_sample"]:
                print(f"    unexpected sample: {audit['unexpected_checkpoint_keys_sample']}")
            if audit["shape_mismatches_sample"]:
                print(f"    shape mismatch sample: {audit['shape_mismatches_sample']}")


def prediction_stats(values: np.ndarray) -> dict:
    arr = values.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_frac": 0.0}
    dy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    dx = np.abs(np.diff(arr, axis=1)).mean() if arr.ndim >= 2 and arr.shape[1] > 1 else 0.0
    return {
        "shape": list(arr.shape),
        "finite_frac": float(np.isfinite(arr).mean()),
        "min": float(np.min(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "roughness_absdiff_mean": float((dx + dy) * 0.5),
    }


def log_grid_to_wandb(
    args: argparse.Namespace,
    output_path: Path,
    metadata: dict,
) -> None:
    if not args.wandb_project:
        return

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "wandb is not installed in this environment. Install it or omit "
            "--wandb_project."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or output_path.stem,
        group=args.wandb_group,
        config={
            "corruptions_root": metadata["corruptions_root"],
            "severity": metadata["severity"],
            "reference_rel": metadata["reference_rel"],
            "models": metadata["models"],
            "output": str(output_path),
            "cmap": args.cmap,
            "normalize_low": args.normalize_low,
            "normalize_high": args.normalize_high,
        },
    )

    caption = args.caption or (
        f"SCARED corruptions | severity={metadata['severity']} | "
        f"frame={metadata['reference_rel']}"
    )
    wandb.log({args.wandb_key: wandb.Image(str(output_path), caption=caption)})
    wandb.save(str(output_path))
    run.finish()


def main() -> None:
    args = parse_args()
    corruptions_root = Path(args.corruptions_root).expanduser()
    output_path = Path(args.output).expanduser()
    exts = extension_list(args.extensions)
    specs = load_model_specs(args)
    corruptions = select_corruptions(corruptions_root, args.corruptions)

    if not corruptions:
        raise RuntimeError("No corruptions selected")

    first_data_root = data_root_for_severity(
        severity_root(corruptions_root, corruptions[0], args.severity),
        args.nested_data_dir,
    )
    _, reference_rel = pick_reference_image(
        root=first_data_root,
        rel_image=args.rel_image,
        split_file=args.split_file,
        split_index=args.split_index,
        exts=exts,
    )

    predictors = build_predictors(args, specs)
    print_model_summary(specs)
    if args.print_load_audit:
        print_load_audit(specs)

    cell_size = (args.cell_width, args.cell_height)
    header_font = load_font(args.font_size, bold=True)
    label_font = load_font(args.label_font_size, bold=True)
    small_font = load_font(max(10, args.label_font_size - 1), bold=False)

    columns = ["Input"] + [spec.name for spec in specs]
    n_cols = len(columns)
    n_rows = len(corruptions)
    gap = args.gap
    caption_h = 0
    if args.caption:
        caption_h = max(28, args.font_size + 14)
    canvas_w = n_cols * args.cell_width + (n_cols - 1) * gap
    canvas_h = (
        args.header_height
        + gap
        + n_rows * args.cell_height
        + (n_rows - 1) * gap
        + caption_h
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    header_y = 0
    for col_idx, label in enumerate(columns):
        x = col_idx * (args.cell_width + gap)
        color = (88, 113, 232) if col_idx == 0 else (231, 91, 69)
        draw_header(canvas, x, header_y, args.cell_width, args.header_height, label, header_font, color)

    metadata = {
        "corruptions_root": str(corruptions_root),
        "severity": args.severity,
        "reference_rel": reference_rel,
        "models": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "path": str(spec.path),
                "code_root": str(spec.code_root) if spec.code_root is not None else None,
                "pretrained_path": str(spec.pretrained_path) if spec.pretrained_path is not None else None,
                "checkpoint_files": spec.checkpoint_files or [],
                "load_audit": spec.load_audit or [],
            }
            for spec in specs
        ],
        "rows": [],
    }

    for row_idx, corruption in enumerate(corruptions):
        y = args.header_height + gap + row_idx * (args.cell_height + gap)
        sev_root = severity_root(corruptions_root, corruption, args.severity)
        data_root = data_root_for_severity(sev_root, args.nested_data_dir)
        image_path, row_rel = resolve_row_image(
            data_root=data_root,
            rel_path=reference_rel,
            rel_image=args.rel_image,
            split_file=args.split_file,
            split_index=args.split_index,
            exts=exts,
        )
        rgb = Image.open(image_path).convert("RGB")
        input_cell = resize_to_cell(rgb, cell_size)
        input_cell = draw_row_label(input_cell, display_label(corruption), label_font)
        canvas.paste(input_cell, (0, y))

        row_meta = {
            "corruption": corruption,
            "image_path": str(image_path),
            "relative_path": row_rel,
            "cells": [],
        }

        for col_idx, spec in enumerate(specs, start=1):
            x = col_idx * (args.cell_width + gap)
            try:
                if spec.kind != "predictions":
                    if isinstance(predictors[spec.name], Exception):
                        raise RuntimeError(str(predictors[spec.name]))
                    pred = predictors[spec.name].predict(rgb)
                    stats = prediction_stats(pred)
                    if args.print_prediction_stats and row_idx < args.prediction_stats_rows:
                        print(
                            "[PRED] {corr} | {model}: p1={p1:.4g} p50={p50:.4g} "
                            "p99={p99:.4g} std={std:.4g} rough={rough:.4g}".format(
                                corr=corruption,
                                model=spec.name,
                                p1=stats.get("p1", float("nan")),
                                p50=stats.get("p50", float("nan")),
                                p99=stats.get("p99", float("nan")),
                                std=stats.get("std", float("nan")),
                                rough=stats.get("roughness_absdiff_mean", float("nan")),
                            )
                        )
                    cell = prediction_to_image(
                        pred,
                        cell_size=cell_size,
                        cmap=args.cmap,
                        low=args.normalize_low,
                        high=args.normalize_high,
                        invert=False,
                    )
                    row_meta["cells"].append(
                        {"model": spec.name, "source": "inference", "prediction_stats": stats}
                    )
                else:
                    pred_path = find_prediction(spec.path, corruption, args.severity, row_rel)
                    if pred_path is None:
                        raise FileNotFoundError(
                            f"No prediction found for {corruption}/{row_rel} in {spec.path}"
                        )
                    loaded = load_prediction_file(pred_path)
                    if isinstance(loaded, Image.Image):
                        cell = resize_to_cell(loaded, cell_size)
                        stats = None
                    else:
                        stats = prediction_stats(loaded)
                        cell = prediction_to_image(
                            loaded,
                            cell_size=cell_size,
                            cmap=args.cmap,
                            low=args.normalize_low,
                            high=args.normalize_high,
                            invert=args.invert_prediction_files,
                        )
                    row_meta["cells"].append(
                        {"model": spec.name, "source": str(pred_path), "prediction_stats": stats}
                    )
            except Exception as exc:
                if args.missing_policy == "error":
                    raise
                cell = placeholder_cell(str(exc), cell_size, small_font)
                row_meta["cells"].append({"model": spec.name, "error": str(exc)})
            canvas.paste(cell, (x, y))

        metadata["rows"].append(row_meta)

    if args.caption:
        draw = ImageDraw.Draw(canvas)
        y = canvas_h - caption_h + 6
        draw.text((0, y), args.caption, fill=(30, 30, 30), font=header_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    if args.save_metadata:
        meta_path = output_path.with_suffix(output_path.suffix + ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {meta_path}")

    log_grid_to_wandb(args, output_path, metadata)
    print(f"Saved grid: {output_path}")


if __name__ == "__main__":
    main()
