import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor
from model import TransformerModel

def load_audio(file_path, sample_rate=16000):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        if not isinstance(sr, int):
            sr = int(sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def preprocess_audio(audio, sample_rate=16000, segment_duration=10):
    segment_length = segment_duration * sample_rate
    if len(audio) > segment_length:
        audio_segment = audio[-segment_length:]
    else:
        # Pad the audio with zeros if it is shorter than the segment length
        padding_length = segment_length - len(audio)
        audio_segment = np.pad(audio, (padding_length, 0), 'constant')
    return audio_segment

def infer(model, processor, audio_segment, device):
    input_values = processor(audio_segment, return_tensors='pt', sampling_rate=16000).input_values.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_values).squeeze(-1)
        probabilities = torch.sigmoid(logits)
    return probabilities.item()

def main(audio_file_path, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(model_path))
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

    audio, sr = load_audio(audio_file_path)
    if audio is None:
        print("Error loading audio file.")
        return
    audio_segment = preprocess_audio(audio, sr)

    probability = infer(model, processor, audio_segment, device)
    print(f"Probability of speaker change at the end of the segment: {probability:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for speaker change detection")
    parser.add_argument("audio_file_path", type=str, help="Path to the audio file")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model")
    args = parser.parse_args()
    main(args.audio_file_path, args.model_path)
