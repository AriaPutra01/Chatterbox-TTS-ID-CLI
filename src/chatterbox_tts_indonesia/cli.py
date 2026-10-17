import warnings, sys, argparse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path

from chatterbox_tts_indonesia.device import detect_device
from chatterbox_tts_indonesia.tts import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MODEL_REPO,
    load_model,
    save_audio,
    synthesize,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="chatterbox-tts-indonesia",
        description="Chatterbox TTS Indonesian CLI Tool - Text-to-Speech & Voice Cloning",
    )

    # Input text options
    text_group = parser.add_argument_group("Text Input")
    text_group.add_argument(
        "-t",
        "--text",
        type=str,
        help="Input text to synthesize (Bahasa Indonesia).",
    )
    text_group.add_argument(
        "-f",
        "--text-file",
        type=Path,
        help="Path to text file to synthesize.",
    )
    text_group.add_argument(
        "positional_text",
        nargs="?",
        type=str,
        help="Optional positional text if --text or --text-file is not specified.",
    )

    # Audio prompt / sound options
    audio_group = parser.add_argument_group("Voice / Audio Prompt")
    audio_group.add_argument(
        "-a",
        "--audio-prompt",
        "--sound",
        "--voice",
        dest="audio_prompt",
        type=Path,
        default=None,
        help="Path to reference audio file (.wav) for voice cloning.",
    )

    # Output options
    output_group = parser.add_argument_group("Output Target")
    output_group.add_argument(
        "-o",
        "--output",
        "--target",
        dest="output",
        type=str,
        default="output.wav",
        help="Target output file (e.g. hasil.wav) or target directory (e.g. ./output/). Defaults to output.wav.",
    )

    # Hardware & Model options
    config_group = parser.add_argument_group("Hardware & Model Options")
    config_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (auto: auto-detect cuda -> mps -> cpu).",
    )
    config_group.add_argument(
        "--model-repo",
        type=str,
        default=DEFAULT_MODEL_REPO,
        help=f"HuggingFace model repository ID (default: {DEFAULT_MODEL_REPO}).",
    )
    config_group.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_FILENAME,
        help=f"T3 checkpoint filename (default: {DEFAULT_CHECKPOINT_FILENAME}).",
    )
    config_group.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration factor (default: 0.5).",
    )
    config_group.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    config_group.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Classifier-Free Guidance weight (default: 0.5).",
    )

    return parser.parse_args(args)


def resolve_input_text(args) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        if not args.text_file.exists():
            sys.exit(f"Error: Text file not found at '{args.text_file}'")
        return args.text_file.read_text(encoding="utf-8").strip()
    if args.positional_text:
        return args.positional_text.strip()

    # Interactive prompt if stdin is a terminal
    if sys.stdin.isatty():
        try:
            user_input = input("Masukkan teks Bahasa Indonesia: ").strip()
            if user_input:
                return user_input
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nOperasi dibatalkan.")

    sys.exit(
        "Error: Input text required. Provide -t 'teks', -f file.txt, or positional text."
    )


def main(argv=None):
    args = parse_args(argv)
    text = resolve_input_text(args)

    if args.audio_prompt and not args.audio_prompt.exists():
        sys.exit(f"Error: Audio prompt file not found at '{args.audio_prompt}'")

    print(f"Device target: {args.device} (resolved: {detect_device(args.device)})")
    print(f"Input text   : {text[:60]}{'...' if len(text) > 60 else ''}")
    if args.audio_prompt:
        print(f"Audio prompt : {args.audio_prompt}")

    model = load_model(
        device=args.device,
        model_repo=args.model_repo,
        checkpoint_filename=args.checkpoint,
    )

    print("Generating speech...")
    wav_tensor, sr = synthesize(
        model=model,
        text=text,
        audio_prompt_path=args.audio_prompt,
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
    )

    out_file = save_audio(wav_tensor, sr, args.output)
    print(f"Berhasil disimpan: {out_file} (sample rate: {sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
