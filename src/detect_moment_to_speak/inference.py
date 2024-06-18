import torch
import numpy as np
from transformers import Wav2Vec2Processor
from model import TransformerModel
import soundfile as sf
import librosa

def infer(audio_path):
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = TransformerModel()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    input_values = processor(audio, return_tensors='pt', padding=True, sampling_rate=sr).input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values).squeeze(-1)
    
    predictions = torch.sigmoid(outputs).cpu().numpy()
    return predictions

if __name__ == "__main__":
    predictions = infer('response.mp3')
    print(predictions)
