# dia2_node.py (seed-enabled)
from __future__ import annotations
import os
import sys
import gc
import json
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
import soundfile as sf

import folder_paths

# Ensure the toolbox folder is on sys.path so the bundled dia2 package imports work
toolbox_dir = os.path.dirname(__file__)
if toolbox_dir not in sys.path:
    sys.path.append(toolbox_dir)

try:
    from dia2 import GenerationConfig, SamplingConfig
    from dia2.engine import Dia2
except Exception:
    Dia2 = None
    GenerationConfig = None
    SamplingConfig = None

CATEGORY = "lethris🧠/Dia2"


def _collect_safetensors_roots() -> list[str]:
    """Collect safetensors/weights files from configured model roots (dropdown values)."""
    out = []

    # 1️⃣ Look in the correct Dia2 folder first
    try:
        files = folder_paths.get_filename_list("dia2") or []
    except Exception:
        files = []

    for f in files:
        if str(f).lower().endswith((".safetensors", ".pt", ".pth", ".bin")):
            out.append(f)

    # Remove duplicates
    out = sorted(dict.fromkeys(out))

    # 2️⃣ If nothing found, check diffusion_models *only to warn the user*
    try:
        alt_files = folder_paths.get_filename_list("diffusion_models") or []
    except Exception:
        alt_files = []

    alt_files = [f for f in alt_files if str(f).lower().endswith((".safetensors", ".pt", ".pth", ".bin"))]

    if alt_files:
        print(
            "[Dia2] ⚠ Warning: Dia2 weights detected in 'diffusion_models'. "
            "Please move them to the 'dia2' folder for proper usage."
        )
        out.extend(alt_files)

    return out or [""]


def _collect_tokenizer_candidates() -> list[str]:
    out = []
    for key in ("dia2", "diffusion_models"):
        try:
            files = folder_paths.get_filename_list(key) or []
        except Exception:
            files = []
        for f in files:
            fn = str(f).lower()
            if "tokenizer" in fn and fn.endswith((".json", ".txt")):
                out.append(f)
    out = sorted(dict.fromkeys(out))
    out.insert(0, "")
    return out


def _audio_input_to_tempfile(audio_obj: Optional[dict], name_prefix: str = "prefix") -> Optional[str]:
    if not audio_obj:
        return None
    try:
        wf = audio_obj.get("waveform", None)
        sr = audio_obj.get("sample_rate", None)
        if wf is None or sr is None:
            return None
        if isinstance(wf, torch.Tensor):
            t = wf.detach().cpu().float()
            if t.ndim == 3:
                t = t[0]
            elif t.ndim == 1:
                t = t.unsqueeze(0)
            arr = t.numpy()
        else:
            arr = np.asarray(wf)
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]
        if arr.shape[0] > 1:
            arr_mono = arr.mean(axis=0)
        else:
            arr_mono = arr[0]
        arr_mono = arr_mono.astype(np.float32)
        tmp_dir = folder_paths.get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{name_prefix}.wav")
        sf.write(out_path, arr_mono, int(sr))
        return out_path
    except Exception:
        return None


class Dia2_TTS_Generator:
    NODE_COLOR = "#A05BFF"  # 💜 audio purple

    @classmethod
    def INPUT_TYPES(cls):
        model_opts = _collect_safetensors_roots()
        tokenizer_opts = _collect_tokenizer_candidates()

        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "[S1] Welcome to the Dia2 TTS Generator.\n"
                            "[S2] Nice. Are we testing voices today?\n"
                            "[S1] Yep. Just a quick demo. (laughs).\n"
                            "[S2] Great. I hope I sound normal this time.\n"
                            "[S1] No promises. Press Generate and find out."
                        )
                    }
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "model_file": (model_opts,),
                "tokenizer_file": (tokenizer_opts,),
                "mimi_id": ("STRING", {"default": ""}),
                "device": (["auto", "cuda", "cpu"],),
                "dtype": (["auto", "bfloat16", "float16", "float32"],),
                "use_cuda_graph": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": True}),
                "output_format": (["mp3", "flac", "wav"],),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0}),
                "text_temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "text_top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "audio_temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "audio_top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
            },
            "optional": {
                "Voice_Sample_S1": ("AUDIO", {}),
                "Voice_Sample_S2": ("AUDIO", {}),
            },
        }

    # 🚫 Removed log output
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "timestamps_json")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        prompt: str,
        seed: int,
        model_file: str,
        tokenizer_file: str,
        mimi_id: str,
        device: str,
        dtype: str,
        use_cuda_graph: bool,
        verbose: bool,
        save_output: bool,
        output_format: str,
        cfg_scale: float,
        text_temperature: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_k: int,
        Voice_Sample_S1: Optional[dict] = None,
        Voice_Sample_S2: Optional[dict] = None,
    ):
        # ---------- Set seeds ----------
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ---------- Prep model paths ----------
        try:
            device_use = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
            dtype_use = "bfloat16" if (dtype == "auto" and device_use == "cuda") else ("float32" if dtype == "auto" else dtype)
            if not model_file:
                raise FileNotFoundError("No Dia2 weights file selected.")
            model_path = Path(model_file).resolve() if os.path.exists(model_file) else Path(folder_paths.get_full_path("dia2", model_file)).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Dia2 weights not found: {model_file}")
            if model_path.is_file():
                weights_path = str(model_path)
                model_dir = model_path.parent
            else:
                model_dir = model_path
                found = next((f for f in model_dir.iterdir() if f.name.lower().endswith((".safetensors", ".pt", ".pth"))), None)
                if found is None:
                    raise FileNotFoundError(f"No weights file found under: {model_dir}")
                weights_path = str(found)
            config_candidates = ["config.json", "dia2-2B.json"]
            config_path = next((model_dir / c for c in config_candidates if (model_dir / c).exists()), None)
            if config_path is None:
                config_path = next((f for f in model_dir.iterdir() if f.suffix.lower() == ".json" and "config" in f.name.lower()), None)
            if config_path is None:
                raise FileNotFoundError(f"No Dia2 config found in model folder: {model_dir}")
            tokenizer_id: Optional[str] = None
            if tokenizer_file and tokenizer_file.strip():
                resolved = Path(tokenizer_file).resolve() if os.path.exists(tokenizer_file) else Path(folder_paths.get_full_path("dia2", tokenizer_file)).resolve()
                tokenizer_id = str(resolved.parent)
            mimi_val = mimi_id.strip() if mimi_id else None
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Dia2: Preparation failed: {e}\n\n{tb}")

        # ---------- Load Dia2 ----------
        try:
            if Dia2 is None:
                raise RuntimeError("Dia2 package not available.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dia = Dia2.from_local(
                config_path=str(config_path),
                weights_path=str(weights_path),
                device=device_use,
                dtype=dtype_use,
                tokenizer_id=tokenizer_id,
                mimi_id=mimi_val,
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Dia2 load failed: {e}\n\n{tb}")

        gen_cfg = GenerationConfig(
            cfg_scale=float(cfg_scale),
            text=SamplingConfig(temperature=float(text_temperature), top_k=int(text_top_k)),
            audio=SamplingConfig(temperature=float(audio_temperature), top_k=int(audio_top_k)),
            use_cuda_graph=bool(use_cuda_graph),
        )

        kwargs: dict[str, Any] = {}
        prefix1_path = _audio_input_to_tempfile(Voice_Sample_S1, "dia2_Voice_Sample_S1") if Voice_Sample_S1 else None
        prefix2_path = _audio_input_to_tempfile(Voice_Sample_S2, "dia2_Voice_Sample_S2") if Voice_Sample_S2 else None
        if prefix1_path:
            kwargs["prefix_speaker_1"] = prefix1_path
        if prefix2_path:
            kwargs["prefix_speaker_2"] = prefix2_path

        try:
            result = dia.generate(prompt, config=gen_cfg, verbose=bool(verbose), **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Dia2 generation failed: {e}\n\n{tb}")

        wave_t = getattr(result, "waveform", None)
        if wave_t is None:
            wave_t = getattr(result, "audio", None)

        if wave_t is None:
            raise RuntimeError("No waveform found in Dia2 result.")

        waveform_tensor = torch.tensor(np.asarray(wave_t), dtype=torch.float32) if not isinstance(wave_t, torch.Tensor) else wave_t.detach().cpu().float()
        sample_rate = getattr(result, "sample_rate", 44100)

        timestamps = getattr(result, "timestamps", None)
        timestamps_json = json.dumps([[w, float(t)] for (w, t) in timestamps]) if timestamps else "[]"

        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
        elif waveform_tensor.ndim == 2:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.shape[1] > 2:
            waveform_tensor = waveform_tensor.mean(dim=1, keepdim=True)

        if save_output:
            try:
                out_dir = folder_paths.get_output_directory() or str(model_dir)
                os.makedirs(out_dir, exist_ok=True)
                base_name = f"dia2_{Path(weights_path).stem}"
                out_path = Path(out_dir) / f"{base_name}.{output_format}"
                arr_to_write = waveform_tensor.squeeze(0).cpu().numpy()
                if arr_to_write.ndim == 2 and arr_to_write.shape[0] > 1:
                    arr_to_write = arr_to_write.T
                elif arr_to_write.ndim == 1:
                    pass
                else:
                    arr_to_write = arr_to_write.squeeze()
                if output_format in ("flac", "wav"):
                    sf.write(str(out_path), arr_to_write, int(sample_rate))
            except Exception:
                pass

        audio_obj = {"waveform": waveform_tensor, "sample_rate": int(sample_rate)}

        return (audio_obj, timestamps_json)


# Register node
NODE_CLASS_MAPPINGS = {"Dia2_TTS_Generator": Dia2_TTS_Generator}
NODE_DISPLAY_NAME_MAPPINGS = {"Dia2_TTS_Generator": "💬 Dia2-2B TTS (Lethris)"}
