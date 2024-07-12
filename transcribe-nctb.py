import os
from glob import glob
from tqdm import tqdm
import torch
import whisperx
import torchaudio

device = "cuda" 
compute_type = "float32" 
language = "bn"
batch_size = 16

model = whisperx.load_model("intelsense/bengali-whisper-medium-tugstugi-ct2", device, compute_type=compute_type, language=language, threads=4)

audio_paths = glob("/home/ubuntu/nctb-cropped/audio/*.wav")

transcriptions = []
for audio_path in tqdm(audio_paths):
  audio, sr = torchaudio.load(audio_path)
  transcription = model.transcribe(audio[0].numpy(), batch_size=batch_size)
  text = " ".join(seg["text"] for seg in transcription["segments"])
  transcriptions.append(text)

with open("transcriptions.txt", "w") as f:
  for audio_path, text in zip(audio_paths, transcriptions):
    f.write(f"{audio_path}|{text}\n")
