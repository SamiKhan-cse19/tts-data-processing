import torch
import whisperx
import torchaudio
from pyannote.audio import Pipeline, Model, Inference

from cleanunet import CleanUNet

import chromadb

import numpy as np

import os
from glob import glob
os.makedirs("audio", exist_ok=True)


device = "cuda" 
compute_type = "float32" 
language = "bn"
batch_size = 8
min_chunk_size = 5
max_chunk_size = 10

model = whisperx.load_model("intelsense/bengali-whisper-medium-tugstugi-ct2", device, compute_type=compute_type, language=language, threads=4)
model_a, metadata = whisperx.load_align_model(model_name="intelsense/wav2vec2_BanglaASR_better", language_code=language, device=device)
# diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device))
diarization_pipeline.embedding_exclude_overlap = True
embedding_model = Model.from_pretrained("pyannote/embedding").to(torch.device(device))
inference = Inference(embedding_model, window="whole")
net = CleanUNet.from_pretrained(varient='full', device=device)


client = chromadb.PersistentClient(path="chroma.db")
collection = client.get_or_create_collection(
    name="speakers-collection", 
    metadata={"hnsw:space": "cosine"}
)


def get_speaker_id(embedding, threshold=0.5):
    embedding = embedding.astype(np.float_).tolist()
    result = collection.query([embedding], n_results=1, include=['embeddings', 'distances'])
    speaker_id = None
    if len(result["ids"][0])>0 and result["distances"][0][0] < threshold:
        speaker_id = result["ids"][0][0]
        # update embedding with running average
        prev_embedding = np.asarray(result["embeddings"][0][0])
        new_embedding = np.asarray(embedding)
        avg_embedding = prev_embedding * 0.9 + new_embedding * 0.1
        avg_embedding = avg_embedding.astype(np.float_).tolist()
        collection.update(
            ids=[speaker_id],
            embeddings=[avg_embedding]
        )
    else:
        # create new speaker and assign new id
        speaker_id = f"sp{collection.count()}"
        collection.add(
            embeddings=[embedding],
            ids=[speaker_id]
        )
    return speaker_id

def chunk_audio(audio, chunk_size):
    num_chunks = int(np.ceil(audio.shape[1] / chunk_size))
    return [audio[:, i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


meta = []

for audio_file in glob('/home/ubuntu/samples/*.mp3'):
    print(audio_file)
    audio_file_id = os.path.basename(audio_file).split('.')[0]
    idx = 0
    waveform, sample_rate = torchaudio.load(audio_file)
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    diarization = diarization_pipeline(audio)
    torch.cuda.empty_cache()

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg_start = turn.start
        seg_end = turn.end

        if seg_end - seg_start < min_chunk_size:
            continue

        seg_audio = waveform[:, int(seg_start * sample_rate):int(seg_end * sample_rate)]

        speaker_embedding = inference({"waveform": seg_audio, "sample_rate": sample_rate})
        torch.cuda.empty_cache()

        speaker_id = get_speaker_id(speaker_embedding)
        
        chunk_size_samples = max_chunk_size * sample_rate
        audio_chunks = chunk_audio(seg_audio, chunk_size_samples)
        
        for chunk in audio_chunks:

            if chunk.shape[1] < min_chunk_size * sample_rate:
                continue

            clean_audio = net(chunk.to(device))[0].detach().cpu()
            torch.cuda.empty_cache()

            transcription = model.transcribe(clean_audio[0].numpy(), batch_size=batch_size, chunk_size=max_chunk_size)
            aligned_transcription = whisperx.align(transcription["segments"], model_a,
                                               metadata, clean_audio[0].numpy(), device, return_char_alignments=False)
            torch.cuda.empty_cache()
            for segment in aligned_transcription["segments"]:
                segment_id = f"{audio_file_id}#{idx}_{speaker_id}.mp3"
                idx += 1
                start_time = segment["start"]
                end_time = segment["end"]

                if end_time - start_time < min_chunk_size:
                    continue

                text = segment["text"]
                torchaudio.save(
                    "audio/"+segment_id,
                    clean_audio[:, int(start_time * sample_rate):int(end_time * sample_rate)],
                    sample_rate,
                    format="mp3"
                )
                meta.append({
                    "id": segment_id,
                    "text": text
                })
    print(f"transcriptions completed for {audio_file}")

with open("metadata.txt", "w") as f:
    for item in meta:
        f.write(f"{item['id']}|{item['text']}\n")


print(f"Total {len(meta)} transcriptions completed")
print(f"Total {collection.count()} speakers found")