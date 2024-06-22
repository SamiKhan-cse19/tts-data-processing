import torch
import whisperx
from pyannote.audio import Model, Inference
import torchaudio
from cleanunet import CleanUNet
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
language = "bn"

model_a, metadata = whisperx.load_align_model(model_name="intelsense/wav2vec2_BanglaASR_better", language_code=language, device=device)
embedding_model = Model.from_pretrained("pyannote/embedding").to(torch.device(device))
inference = Inference(embedding_model, window="whole")
net = CleanUNet.from_pretrained(varient='full', device=device)

sent_end_puncts = ['।', '?']
os.makedirs('nctb-clean', exist_ok=True)


# Function to get or create a speaker ID
def get_speaker_id(embedding, threshold=0.5):

    if speakers:
        distances = [cdist(embedding, sp["embedding"], metric="cosine")[0,0] for sp in speakers]
        min_distance = np.min(distances)

        if min_distance < threshold:
            speaker_id = speakers[np.argmin(distances)]['id']
        else:
            speaker_id = len(speakers) + 1
            speakers.append({'id': speaker_id, 'embedding': embedding})
    else:
        speaker_id = 1
        speakers.append({'id': speaker_id, 'embedding': embedding})
    return f"sp{speaker_id}"



def clean_and_segment_audio(audio_file, text):
    audio, sr = torchaudio.load(audio_file)
    duration = len(audio[0])/sr
    if duration > 200: # tackle out of memory error
        return None
    transcription = [{'text':text, 'start':0.0, 'end':duration}]
    aligned_transcription = whisperx.align(transcription, model_a, metadata, audio[0].numpy(), device, return_char_alignments=False)
    
    idx = 0 # segment index for each audio file
    sent_start = None
    sent_end = None
    sent_text = ""    
    segments = []

    try:

        for word in aligned_transcription['segments'][0]['words']:
            if 'score' not in word.keys():
                continue

            if sent_start == None:
                if word['score'] < 0.5:
                    continue
                sent_start = word['start']
            
            if word['word'][-1] in sent_end_puncts or (word['score'] < 0.5 and sent_end != None): 
                if word['score'] > 0.5: 
                    if sent_start == None:
                        sent_start = word['start']
                    sent_end = word['end']
                    sent_text += word['word'] + " "
                else:
                    continue
                
                if sent_end - sent_start < 3: # combine small sentences with the next one
                    continue
                
                seg_audio = audio[:, int(sent_start*sr):int(sent_end*sr)]
                clean_audio = net(seg_audio.to(device))[0].detach().cpu()
                
                torchaudio.save('temp.wav', clean_audio, sr)            
                embedding = inference('temp.wav')
                embedding = embedding.reshape(1,-1)
                # Get or create speaker ID
                speaker_id = get_speaker_id(embedding)

                seg_id = f"{audio_file.split('/')[-1].split('.')[0]}#{str(idx)}_{speaker_id}.wav"
                idx += 1

                torchaudio.save('nctb-clean/'+seg_id, clean_audio, sr)
                segments.append({'id': seg_id, 'text': sent_text.strip()})
                sent_start = None
                sent_text = ""
                
                if word['score'] < 0.5:
                    continue
            else:
                sent_end = word['end']
                sent_text += word['word']+" "
    except:
        print("Error in:", audio_file)
        return None
            
    return segments
    




speakers = [] # for storing and checking speaker embeddings
meta = []
raw = []
with open('/home/ubuntu/nctb-dataset/metadata.txt', 'r') as f:
    for line in f:
        raw.append(line.strip())

for line in tqdm(raw):
    audio_file, text = line.split('|')
    segments = clean_and_segment_audio('/home/ubuntu/nctb-dataset/audio/'+audio_file, text)
    if segments:
        meta = meta + segments

with open('metadata.txt', 'w') as f:
    for file in meta:
        f.write(f"{file['id']}|{file['text']}\n")