import torch
import numpy as np
from transformers import Wav2Vec2Processor
from model import TransformerModel
import soundfile as sf

def infer(audio_path):
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = TransformerModel()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    audio, sr = sf.read(audio_path)
    input_values = processor(audio, return_tensors='pt', padding=True, sampling_rate=sr).input_values

    with torch.no_grad():
        outputs = model(input_values).squeeze(-1)
    
    predictions = torch.sigmoid(outputs).numpy()
    return predictions

if __name__ == "__main__":
    predictions = infer('path_to_audio_file.ogg')
    print(predictions)
