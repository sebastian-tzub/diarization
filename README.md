# Speaker Diarization

Speaker-attributed transcription: take a video/audio file (e.g. match
commentary) and produce a transcript labeled with **who** is speaking and
**what** they said.

Pipeline: `ffmpeg` extract → `pyannote` diarize → `whisper` transcribe → merge
by time overlap.

## Setup

### 1. System dependency: ffmpeg

`ffmpeg` must be on your PATH (used to extract audio and by whisper).

```bash
# Debian/Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

### 2. Python environment

Requires **Python 3.11**. Create a virtualenv and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# GPU machine (NVIDIA, CUDA 12.x):
pip install -r requirements.txt

# CPU-only machine:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The pipeline auto-detects CUDA and falls back to CPU. On CPU, prefer a smaller
whisper model (`--model medium`) — `large-v3` is very slow without a GPU.

### 3. Hugging Face token

The diarization model is gated. One-time setup:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the license at
   <https://huggingface.co/pyannote/speaker-diarization-3.1>.
3. Copy the env template and paste your token in:

   ```bash
   cp .env.example .env
   # edit .env and set HF_TOKEN=hf_...
   ```

`.env` is git-ignored — never commit your token. Scripts also accept
`--hf-token` on the command line, which overrides the env value.

## Usage

```bash
# Minimal — token read from .env:
python diarization_pyannote.py match.mp4

# All options:
python diarization_pyannote.py match.mp4 \
  --num-speakers 2 \
  --model large-v3 \
  --out transcript.txt \
  --hf-token hf_xxxxxxxxxxxx     # optional; defaults to HF_TOKEN from .env
```

`--num-speakers` is optional but recommended — passing the known count (e.g. `2`
for a two-person commentary booth) makes diarization significantly more
accurate.

### Quick test

`test/scripts/test.py` runs the diarize → transcribe → merge stages on the
bundled sample (`test/dataset/test/test.wav`) and writes
`test/dataset/test/output.txt`:

```bash
python test/scripts/test.py
```

### Alternative: WhisperX

`diarization_whisperx.py` is an alternative pipeline built on
[WhisperX](https://github.com/m-bain/whisperX). It does word-level forced
alignment before diarization, giving tighter speaker boundaries. Same token
setup and CLI conventions:

```bash
python diarization_whisperx.py test/dataset/world_cup/world_cup.mp4 --num-speakers 2 --out wx_transcript.txt
```

## Notes

- Football match recordings have loud crowd noise, so verifying transcription
  accuracy on a sample beforehand is essential.
- pyannote.audio 4.x pulls in `torchcodec`, which can't load the audio when the
  system has FFmpeg 8 (it supports 4–7). This pipeline sidesteps that by loading
  audio in-memory with `scipy`, so any `torchcodec`/FFmpeg warnings on startup
  are harmless.
- Whisper decodes in fp32 (`fp16=False`) because fp16 produces NaN logits on
  some GPUs.
