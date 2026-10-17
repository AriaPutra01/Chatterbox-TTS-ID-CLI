from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .device import clear_device_cache, detect_device

DEFAULT_MODEL_REPO = "grandhigh/Chatterbox-TTS-Indonesian"
DEFAULT_CHECKPOINT_FILENAME = "t3_cfg.safetensors"


def load_model(
    device: str = "auto",
    model_repo: str = DEFAULT_MODEL_REPO,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> ChatterboxTTS:
    """Load base Chatterbox and swap T3 weights with Indonesian fine-tune."""
    resolved_device = detect_device(device)
    print(f"Loading Chatterbox TTS on device: {resolved_device}")

    model = ChatterboxTTS.from_pretrained(device=resolved_device)

    try:
        model.t3.tfmr.set_attn_implementation("eager")
    except AttributeError:
        model.t3.tfmr.config._attn_implementation = "eager"

    print(
        f"Downloading Indonesian checkpoint from {model_repo} ({checkpoint_filename})..."
    )
    ckpt_path = hf_hub_download(repo_id=model_repo, filename=checkpoint_filename)
    t3_state = load_file(ckpt_path, device="cpu")
    if "model" in t3_state:
        t3_state = t3_state["model"][0]

    model.t3.load_state_dict(t3_state)
    clear_device_cache(resolved_device)
    return model


def synthesize(
    model: ChatterboxTTS,
    text: str,
    audio_prompt_path: Optional[str | Path] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
) -> Tuple[torch.Tensor, int]:
    """Generate audio waveform from text."""
    prompt_str = str(audio_prompt_path) if audio_prompt_path else None
    wav_audio = model.generate(
        text=text,
        audio_prompt_path=prompt_str,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfg_weight,
    )
    return wav_audio, model.sr


def save_audio(
    wav_tensor: torch.Tensor,
    sample_rate: int,
    output_target: str | Path,
) -> Path:
    """Write audio tensor to target folder or specific file."""
    target = Path(output_target).expanduser().resolve()

    if target.is_dir() or str(output_target).endswith(("/", "\\")):
        target.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = target / f"tts_{timestamp}.wav"
    else:
        if target.suffix.lower() != ".wav":
            target = target.with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        out_path = target

    audio_data = wav_tensor.squeeze().detach().cpu().numpy()
    sf.write(str(out_path), audio_data, sample_rate)
    return out_path
