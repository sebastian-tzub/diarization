"""
Runs the pyannote diarization pipeline on the bundled test.wav and writes a
speaker-labeled transcript next to it.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root is two levels up from this file (test/scripts/test.py). Add it to
# the import path so the top-level pipeline module is importable, and anchor the
# data paths to it so this runs correctly from any working directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diarization_pyannote import diarize, transcribe, assign_speakers, write_transcript

load_dotenv(ROOT / ".env")
hf_token = os.environ["HF_TOKEN"]  # set in .env (see .env.example)

wav = str(ROOT / "test/dataset/test/test.wav")
out = str(ROOT / "test/dataset/test/output.txt")

turns = diarize(wav, hf_token=hf_token, num_speakers=2)
segments = transcribe(wav, model_size="medium")
labeled = assign_speakers(segments, turns)
write_transcript(labeled, out)
