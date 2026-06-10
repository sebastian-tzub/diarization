# Speaker Diarization

Speaker-attributed transcription: take a video/audio file (e.g. match
commentary) and produce a transcript labeled with **who** is speaking and
**what** they said.

Two pipelines are provided. **WhisperX (`diarization_whisperx.py`) is the
recommended one** — it does word-level forced alignment before diarization,
giving noticeably tighter speaker boundaries. A simpler `pyannote` + `whisper`
overlap-merge pipeline (`diarization_pyannote.py`) is kept as an alternative.

Both were run on the same video clip in test/dataset/world_cup; outputs are checked in for comparison:

- `test/dataset/world_cup/wx_transcript.txt` — WhisperX
- `test/dataset/world_cup/pyan_transcript.txt` — pyannote + whisper

The WhisperX output is clearly better: no `UNKNOWN` speakers, cleaner turn
boundaries (the pyannote run merges two speakers into one line and drops
overlapping speech), and more accurate text (e.g. "Otamendi" vs "Iota Mendy").

Format of transcription txt files: [<start>s] <SPEAKER>: <text>

## Setup

### 1. System dependency: ffmpeg

`ffmpeg` must be on your PATH (used to extract audio and by whisper).

### 2. Python environment

Newest python versions are generally not compatible wth dependenices in this project

Requires **Python 3.11**. Create a virtualenv and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# GPU machine (NVIDIA, CUDA 12.x):
pip install -r requirements.txt

# CPU-only machine:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Models & hardware

Both scripts auto-detect CUDA and fall back to CPU; no flag needed.

On CPU, prefer a smaller whisper model (`--model medium` or `small`) — `large-v3`
is very slow without a GPU.

**Out of memory (`CUDA failed with error out of memory`)?** WhisperX loads three
models (transcribe → align → diarize). On a small GPU (≈4 GB) the script already
uses `int8`, caps the batch size at 4, and frees each stage's VRAM before the
next. If you still hit it, drop to a lighter model — `--model medium` or
`--model small` — or force CPU with `CUDA_VISIBLE_DEVICES=""`.

### 3. Hugging Face token

The diarization model is gated. One-time setup:
Copy the env template and paste your token in:

   ```bash
   cp .env.example .env
   # edit .env and set HF_TOKEN=hf_...
   ```

`--hf-token` on the command line, which overrides the env value.

## Usage

```bash
# Minimal — token read from .env:
python diarization_whisperx.py match.mp4

# All options:
python diarization_whisperx.py match.mp4 \
  --num-speakers 2 \
  --model large-v3 \
  --out transcript.txt \
  --hf-token hf_xxxxxxxxxxxx     # optional; defaults to HF_TOKEN from .env
```

`--num-speakers` is optional but recommended — passing the known count (e.g. `2`
for a two-person commentary booth) makes diarization significantly more
accurate. Both scripts share the same CLI.

### Quick test

`test/scripts/test.py` runs the diarize (diarization_pyannote) → transcribe → merge stages on the
bundled sample (`test/dataset/test/test.wav`) and writes
`test/dataset/test/output.txt`:

This test is done on a small, non overlapping voices wav file. Not a good quality test, just makes sure 
dependicies are installed correctly and main pipeline can be executed with diarization script files 

```bash
python test/scripts/test.py
```

### Alternative: pyannote + whisper

`diarization_pyannote.py` is the simpler pipeline: `ffmpeg` extract → `pyannote`
diarize → `whisper` transcribe → merge by time overlap. Same token setup and
CLI. Use it if you want to avoid the WhisperX dependency; speaker boundaries are
coarser since merging is segment-level rather than word-level.

```bash
python diarization_pyannote.py match.mp4 --num-speakers 2 --out transcript.txt
```

## Notes

- Football match recordings have loud crowd noise, so verifying transcription
  accuracy on a sample beforehand is essential.
- pyannote.audio 4.x pulls in `torchcodec`, which can't load the audio when the
  system has FFmpeg 8 (it supports 4–7). This pipeline sidesteps that by loading
  audio in-memory with `scipy`, so any `torchcodec`/FFmpeg warnings on startup
  are harmless.
