# Chatterbox-TTS-Indonesian CLI

Tool CLI Text-to-Speech (TTS) dan Voice Cloning Bahasa Indonesia berbasis model [grandhigh/Chatterbox-TTS-Indonesian](https://huggingface.co/grandhigh/Chatterbox-TTS-Indonesian).

## Fitur

- **Deteksi Hardware Otomatis**: Mendukung `cuda` (NVIDIA GPU), `mps` (Apple Silicon Mac), dan `cpu` tanpa hardcode.
- **Fleksibilitas Input Teks**: Mendukung argument `--text`, file `--text-file`, argumen posisional, atau input interaktif.
- **Voice Cloning**: Mendukung file audio prompt / sampel suara referensi (`--voice` / `--audio-prompt`).
- **Target Fleksibel**: Menyimpan ke file `.wav` tertentu atau direktori target (`--output`).

## Instalasi & Setup

Proyek ini menggunakan `uv`:

```bash
uv sync
```

## Penggunaan CLI

### 1. Sintesis Teks Dasar (Default Voice)

```bash
uv run chatterbox-id -t "Selamat pagi Indonesia, bagaimana kabar Anda hari ini?" -o output.wav
```

Atau langsung dengan teks posisional:

```bash
uv run chatterbox-id "Selamat pagi Indonesia" -o ./hasil.wav
```

### 2. Sintesis dengan Voice Cloning (Audio Prompt)

Gunakan file audio `.wav` referensi untuk meniru karakter dan timbre suara:

```bash
uv run chatterbox-id -t "Teks yang ingin disuarakan" -a /path/to/suara_referensi.wav -o hasil_clone.wav
```

### 3. Membaca Teks dari File

```bash
uv run chatterbox-id -f artikel.txt -o ./output/
```

Jika `-o` berupa folder, file akan disimpan otomatis dengan nama berformat timestamp (contoh: `output/tts_20261017_234500.wav`).

### 4. Pilihan Device

Secara default `--device auto` otomatis mendeteksi perangkat terbaik (`cuda` -> `mps` -> `cpu`). Anda juga dapat memaksakan perangkat:

```bash
uv run chatterbox-id -t "Halo" --device cpu
uv run chatterbox-id -t "Halo" --device mps
uv run chatterbox-id -t "Halo" --device cuda
```

## Penggunaan Sebagai Python Module

```python
from chatterbox_tts_indonesia import load_model, synthesize, save_audio

# Load model (otomatis pilih cuda/mps/cpu)
model = load_model(device="auto")

# Generate
wav_tensor, sr = synthesize(
    model,
    text="Selamat datang di Indonesia.",
    audio_prompt_path="referensi.wav",  # opsional
)

# Simpan
save_audio(wav_tensor, sr, "output.wav")
```
