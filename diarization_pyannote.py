#!/usr/bin/env python3
"""
Speaker-attributed transcription pipeline

Goal: take a video/audio file of match commentary and produce a transcript
labeled with WHO is speaking and WHAT they said.

PIPELINE: 
    [0] EXTRACT   video -> 16kHz mono wav            (ffmpeg)
    [1] DIARIZE   wav   -> (start, end, speaker)      (pyannote)
    [2] TRANSCRIBE wav  -> (start, end, text)         (whisper)
    [3] MERGE     overlap-match text segments to speaker turns
"""

import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import whisper

# Load HF_TOKEN (and any other vars) from a local .env if present.
load_dotenv()

def extract_audio(video_path: str, wav_path: str):
    """Extract audio out of the video as 16kHz mono WAV with ffmpeg."""
    command = [
      'ffmpeg',
      '-y',                  
      '-i', video_path,   
      '-vn',               
      '-ar', '16000',        
      '-ac', '1',            
      '-c:a', 'pcm_s16le',   
      wav_path           
    ]
    try: 
      subprocess.run(command, check=True, capture_output = True)
      print(f"Successfully extracted 16kHz audio to {wav_path}")
    except subprocess.CalledProcessError as e:
      print(f"Error during extraction: {e.stderr.decode()}")
    
def diarize(wav_path: str, hf_token: str, num_speakers: int | None = None):
    """Diarization with pyannote, return speaker turns: a list of (start, end, speaker_label)."""
    pipeline = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      token= hf_token
    )

    if torch.cuda.is_available():
      pipeline.to(torch.device("cuda"))

    # Load the WAV in-memory and hand pyannote a waveform tensor. This bypasses
    # pyannote 4.x's torchcodec-based file reader, which can't load the system
    # FFmpeg 8 libraries (torchcodec 0.7 only supports FFmpeg 4-7).
    sample_rate, samples = wavfile.read(wav_path)
    if samples.ndim == 1:
        samples = samples[:, None]
    # int16 PCM -> float32 in [-1, 1], shaped (channels, time) as pyannote expects
    waveform = torch.from_numpy(samples.T.astype(np.float32) / 32768.0)

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=num_speakers,
    )

    # pyannote 4.x returns a DiarizeOutput; .speaker_diarization is the Annotation.
    # Unpack to a list of (start, end, speaker) tuples.
    speaker_turns = []
    for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        speaker_turns.append((segment.start, segment.end, speaker))

    return speaker_turns

def transcribe(wav_path: str, model_size: str = "medium"):
    """
    Transcription with OpenAI Whisper, return text segments: a list of (start, end, text).

    model_size tradeoff: large-v3/turbo = best accuracy, slow on CPU.
                         medium   = reasonable CPU fallback.
    """
    model = whisper.load_model(model_size)
    # fp16=False forces fp32 decoding; fp16 on some GPUs yields NaN logits.
    result = model.transcribe(wav_path, word_timestamps=True, fp16=False)
    text_segments = [(seg["start"], seg["end"], seg["text"])for seg in result["segments"]]

    return text_segments


def _overlap(seg_start, seg_end, turn_start, turn_end):
    """Seconds of overlap between two time intervals (0 if disjoint)."""
    return max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))

  
def assign_speakers(text_segments, speaker_turns):
    """
    Attach a speaker label to each text segment by time-overlap.

    text_segments: list of (start, end, text)
    speaker_turns: list of (start, end, speaker)
    returns:       list of (start, end, speaker, text)

    Since a text segment spans [t_start, t_end]. Each diarization
    turn also spans a time range with a speaker. The right speaker is the
    turn whose overlap with the text segment is largest.

    Thus we use the following formula: 

    overlap(segment, turn) = max(0, min(seg_end, turn_end)
                                    - max(seg_start, turn_start))

    LIMITATION: Speech overlaps are not handled well
    The second speaker with less overlap will never have his audio transcribed in the 
    final output.
    """
    diarized_transcription = []

    for seg_start, seg_end, text in text_segments:
      best_speaker = "UNKNOWN"
      best_overlap = 0.0

      for turn_start, turn_end, speaker in speaker_turns:
          ov = _overlap(seg_start, seg_end, turn_start, turn_end)
          if ov > best_overlap:
              best_overlap = ov
              best_speaker = speaker

      diarized_transcription.append((seg_start, seg_end, best_speaker, text))

    return diarized_transcription 

def write_transcript(labeled_segments, out_path: str) -> None:
    """Write a readable transcript to output file."""
    with open(out_path, "w") as f:
        for start, end, speaker, text in labeled_segments:
            f.write(f"[{start:.1f}s] {speaker}: {text.strip()}\n")

def main():
    parser = argparse.ArgumentParser(description="Speaker-attributed transcription")
    parser.add_argument("input", help="path to video or audio file")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (defaults to HF_TOKEN env / .env)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="set if known, e.g. 2 for a two-person booth")
    parser.add_argument("--model", default="medium", help="whisper model size")
    parser.add_argument("--out", default="transcript.txt")
    args = parser.parse_args()

    if not args.hf_token:
        parser.error("no Hugging Face token: set HF_TOKEN in .env or pass --hf-token")

    wav = str(Path(args.input).with_suffix(".wav"))

    #PIPELINE
    extract_audio(args.input, wav)            
    speaker_turns = diarize(wav, args.hf_token, args.num_speakers)  
    text_segments = transcribe(wav, args.model)                 
    labeled = assign_speakers(text_segments, speaker_turns)     
    write_transcript(labeled, args.out)

    print(f"Done -> {args.out}")

if __name__ == "__main__":
    main()