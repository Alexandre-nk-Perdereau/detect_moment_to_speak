import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from model import TransformerModel

class AudioDataset(Dataset):
    def __init__(self, segments, labels):
        self.segments = segments
        self.labels = labels
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        return segment, label

def collate_fn(batch):
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = processor(inputs, return_tensors='pt', padding=True, sampling_rate=16000).input_values
    labels = torch.tensor(labels, dtype=torch.float32)
    return inputs, labels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    segments = np.load('output/segments.npy', allow_pickle=True)
    labels = np.load('output/segment_labels.npy', allow_pickle=True)

    train_segments, val_segments, train_labels, val_labels = train_test_split(segments, labels, test_size=0.2, random_state=42)
    
    train_dataset = AudioDataset(train_segments, train_labels)
    val_dataset = AudioDataset(val_segments, val_labels)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    model.train()

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_targets = val_batch
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs).squeeze(-1)
                val_loss += criterion(val_outputs, val_targets).item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stop triggered")
            break

if __name__ == "__main__":
    train()
