#!/usr/bin/env python3
"""
Speaker-attributed transcription with WhisperX (alternative to diarization_pyannote.py).

WhisperX bundles the whole pipeline — transcribe (faster-whisper) -> word-level
forced alignment -> diarize (pyannote) -> assign speakers to words — so it gives
tighter word-level speaker boundaries than the manual overlap-matching approach.

Requires a Hugging Face token with the pyannote/speaker-diarization-3.1 license
accepted. Set HF_TOKEN in .env (see .env.example) or pass --hf-token.
"""

import argparse
import os

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()


def diarize_transcribe(audio_file, hf_token, model_size="large-v3",
                       num_speakers=None, batch_size=16):
    """Run the full WhisperX pipeline and return speaker-labeled segments."""
    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
    else:
        # int8 keeps CPU inference tractable; float16 is not supported on CPU.
        device, compute_type = "cpu", "int8"

    # 1. Transcribe with Whisper (faster-whisper backend).
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align for accurate word-level timestamps.
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    # 3. Diarize (who spoke when) and assign speakers to words.
    # Pin to 3.1 for parity with diarization_pyannote.py (WhisperX otherwise
    # defaults to the newer speaker-diarization-community-1 model).
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        token=hf_token,
        device=device,
    )
    if num_speakers is not None:
        diarize_segments = diarize_model(audio, num_speakers=num_speakers)
    else:
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=4)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result["segments"]


def write_transcript(segments, out_path):
    """Write a readable speaker-labeled transcript to out_path."""
    with open(out_path, "w") as f:
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(f"[{seg['start']:.1f}s] {speaker}: {seg['text'].strip()}\n")


def main():
    parser = argparse.ArgumentParser(description="WhisperX speaker-attributed transcription")
    parser.add_argument("input", help="path to video or audio file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (defaults to HF_TOKEN env / .env)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="set if known, e.g. 2 for a two-person booth")
    parser.add_argument("--model", default="large-v3", help="whisper model size")
    parser.add_argument("--out", default="whisperx_transcript.txt")
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("no Hugging Face token: set HF_TOKEN in .env or pass --hf-token")

    segments = diarize_transcribe(
        args.input, args.hf_token, args.model, args.num_speakers
    )
    write_transcript(segments, args.out)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
