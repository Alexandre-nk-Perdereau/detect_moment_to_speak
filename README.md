# Detect Moment to Speak

## Description
Detect Moment to Speak is a project aimed at fine-tuning a model capable of detecting the ideal moments to speak in a multi-person conversation. Each audio track corresponds to a different person, and the model identifies transitions between the moments when one person stops speaking and another begins.

## Objectives
- Load each person's audio tracks into a given folder.
- Detect speech segments for each audio track.
- Create labels indicating transitions between different speakers.
- Train a model to predict ideal speaking times.
- Apply data augmentation techniques to improve model generalization.

## Environment setup

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation
Clone this repository and install the dependencies with Poetry:

```bash
git clone https://github.com/Alexandre-nk-Perdereau/detect_moment_to_speak.git
cd detect_moment_to_speak
poetry install
```

## Data
The input data consists of separate .ogg tracks for each speaker. The preprocessing script processes these tracks, detects speech segments, and creates labels indicating speaker transitions. The labels are smoothed using a Gaussian kernel to represent the probability of a speaker change.

## Preprocessing
The preprocessing script performs the following steps:
1. Load audio tracks from the specified folder.
2. Detect speech segments using the WebRTC Voice Activity Detector (VAD).
3. Create labels for transitions between speakers, applying Gaussian smoothing to represent transition probabilities.
4. Save the processed segments and labels to disk.

To run the preprocessing:

```bash
poetry run python src/detect_moment_to_speak/preprocessing.py
```

## Training
The training script loads the preprocessed segments and labels, applies random transformations for data augmentation, and trains the model to predict the probability of a speaker change at the end of each segment.

### Data Augmentation
Random transformations applied during training include:
- Adding noise
- Time shifting
- Pitch shifting
- Speed changing

These transformations help to avoid overfitting and improve the model's ability to generalize.

To train the model:

```bash
poetry run python src/detect_moment_to_speak/train.py
```

## Inference
The inference script loads a pre-trained model and processes an input audio file to predict the probability of a speaker change at the end of the last 10 seconds. If the audio is longer than 10 seconds, only the last 10 seconds are used. If it is shorter, it is padded with zeros.

To perform inference:

```bash
poetry run python src/detect_moment_to_speak/inference.py <audio_file_path> <model_path>
```

Replace `<audio_file_path>` with the path to your audio file and `<model_path>` with the path to your trained model.

## Authors
- Alexandre Perdereau

## License
This project is licensed under the MIT License.