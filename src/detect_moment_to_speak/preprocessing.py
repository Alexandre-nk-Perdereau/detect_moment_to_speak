import numpy as np
import librosa
import webrtcvad
import os

def load_audio(file_path, sample_rate=16000):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        if not isinstance(sr, int):
            sr = int(sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def detect_speech(audio, sample_rate, vad_mode=3):
    vad = webrtcvad.Vad(vad_mode)
    frames = librosa.util.frame(audio, frame_length=sample_rate // 100, hop_length=sample_rate // 100).T
    speech_frames = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]
    return np.array(speech_frames)

def process_folder(folder_path, sample_rate=16000):
    tracks = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ogg'):
                file_path = os.path.join(root, file)
                audio, sr = load_audio(file_path, sample_rate)
                if audio is not None and sr is not None:
                    speech_frames = detect_speech(audio, sr)
                    tracks.append((file_path, speech_frames, len(audio)))
    return tracks

def gaussian_kernel(size, sigma=1):
    """Generates a 1D Gaussian kernel."""
    x = np.arange(-size, size + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()

def create_labels_for_tracks(tracks, sample_rate=16000, window_size=50):
    total_length = max([length for _, _, length in tracks])
    combined_labels = np.zeros(total_length // (sample_rate // 100))
    
    for i in range(len(tracks)):
        file_path, speech_frames, _ = tracks[i]
        print(f"Processing {file_path}")
        
        # Combine the speech frames from all tracks
        for j in range(len(speech_frames)):
            if speech_frames[j]:
                combined_labels[j] = i + 1  # Marque la personne qui parle

    # Detect transitions between different speakers and apply Gaussian normalization
    labels = np.zeros(len(combined_labels))
    kernel = gaussian_kernel(window_size)
    for i in range(1, len(combined_labels)):
        if combined_labels[i] != combined_labels[i-1] and combined_labels[i-1] != 0:
            start = max(0, i - window_size)
            end = min(len(combined_labels), i + window_size + 1)
            labels[start:end] += kernel[:end-start]  # Apply the Gaussian kernel
    
    labels = np.clip(labels, 0, 1)  # Ensure labels are between 0 and 1
    return labels


def save_segments_and_labels(audio, sr, labels, segment_duration=10, output_dir='output', step_size=0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    segment_length = segment_duration * sr
    step_length = int(step_size * sr)  # Step size in samples
    segments = []
    segment_labels = []

    for start in range(0, len(audio) - segment_length, step_length):
        end = start + segment_length
        segment = audio[start:end]
        # Utiliser l'étiquette correspondant à la fin du segment
        label_index = (end // (sr // 100)) - 1  # indice pour la fin du segment
        if label_index >= len(labels):
            break  # Éviter d'aller hors limites
        segment_label = labels[label_index]
        segments.append(segment)
        segment_labels.append(segment_label)

    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    np.save(os.path.join(output_dir, 'segments.npy'), segments)
    np.save(os.path.join(output_dir, 'segment_labels.npy'), segment_labels)

def process_all_files(data_dir='data', output_dir='output', segment_duration=10, sample_rate=16000, step_size=0.3):
    for root, dirs, _ in os.walk(data_dir):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            tracks = process_folder(folder_path, sample_rate)
            if tracks:
                combined_audio = np.concatenate([load_audio(file_path, sample_rate)[0] for file_path, _, _ in tracks])
                labels = create_labels_for_tracks(tracks, sample_rate)
                save_segments_and_labels(combined_audio, sample_rate, labels, segment_duration, output_dir, step_size)

if __name__ == "__main__":
    process_all_files()
