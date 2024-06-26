import torch.nn as nn
from transformers import Wav2Vec2Model

class TransformerModel(nn.Module):
    def __init__(self, pretrained_model_name='facebook/wav2vec2-base', num_labels=1):
        super(TransformerModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)
        
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values).last_hidden_state
        logits = self.classifier(outputs[:, -1, :])
        return logits  # We do not apply a sigmoïd here because we use BCEWithLogitsLoss
