# Detect Moment to Speak

## Description
Detect Moment to Speak is a project aimed at fine tuning a model capable of detecting the ideal moments to speak in a multi-person conversation. Each audio track corresponds to a different person, and the model identifies transitions between the moments when one person stops speaking and another begins.

## Objectives
- Load each person's audio tracks into a given folder.
- Detect speech segments for each audio track.
- Create labels indicating transitions between different speakers.
- Train a model to predict ideal speaking times.

## Environment setup

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation
Clone this repository and install the dependencies with Poetry :

```bash
git clone https://github.com/Alexandre-nk-Perdereau/detect_moment_to_speak.git
cd detect_moment_to_speak
poetry install
```

## Data
the input data are separate .ogg tracks for each speaker: this is the output format provided by the discord craig bot.

data/
├── project1/
│   ├── speaker1.ogg
│   ├── speaker2.ogg
│   └── speaker3.ogg
├── project2/
│   ├── speaker1.ogg
│   └── speaker2.ogg
└── ...

## Data pre-processing
The preprocessing script recursively scans data folders, detects speech segments and creates labels for the ideal moments to speak.

### Running Preprocessing
Place your audio projects in the `data/` folder, then run the preprocessing script:

```bash
poetry run python src/detect_moment_to_speak/preprocessing.py
```

## Model training
The training script uses generated audio segments and labels to train a transformer-based model.

### Running the training script
Run the training script with :

```bash
poetry run python src/detect_moment_to_speak/train.py
```

The training uses an early-stop mechanism to automatically determine the optimal number of epochs.

## Inference
The inference script loads a pre-trained model and predicts the ideal times to speak in a given audio file.

### Run Inference
Run the inference script with :

```bash
poetry run python src/detect_moment_to_speak/inference.py
```

## Contributors
- Alexandre Perdereau

## License
This project is licensed under the MIT License.