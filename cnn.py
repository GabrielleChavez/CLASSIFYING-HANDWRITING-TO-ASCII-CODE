from dataset import IAMProcessedDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.optim.lr_scheduler import StepLR

class IAMProcessedDataset(Dataset):

    def __init__(self, lines_path, vocab_path, transform=None):
        self.transform = transform
        self.img_list = []
        for folder in os.listdir(lines_path):
            for inner_folder in os.listdir(os.path.join(lines_path, folder)):
                transcription_file = os.path.join(lines_path, folder, inner_folder, f'{inner_folder}-transcription.txt')
                with open(transcription_file, 'r') as f:
                    transcription = f.read()
                img_name = inner_folder
                img_path = os.path.join(lines_path, folder, inner_folder, f'{inner_folder}.png')
                self.img_list.append((img_name, img_path, transcription))

        self.char_to_idx, self.idx_to_char, self.num_classes, self.blank_idx = get_vocab(vocab_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.img_list[idx][1]
        transcription = self.img_list[idx][2]
        encoded_transcription = encode(transcription, self.char_to_idx)

        
        image = Image.open(image_path).convert("L")
        to_tensor = transforms.Compose([transforms.ToTensor()])
        image = to_tensor(image)

        if self.transform:
            image = self.transform(image)

        sample = (image, encoded_transcription)

        return sample
    
class HandwritingRecognitionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Adaptive pooling for height normalization
            nn.AdaptiveAvgPool2d((1, None))
            # Add more CNN layers as needed
        )
        self.rnn = nn.LSTM(input_size=64, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        cnn_features = self.cnn(x)  # (batch_size, channels, 1, width')
        cnn_features = cnn_features.squeeze(2)  # Remove the height dimension (now 1)
        # cnn_features shape: (batch_size, width', channels)
        rnn_out, _ = self.rnn(cnn_features.permute(0, 2, 1))  # (batch_size, width', hidden_size * 2)
        logits = self.fc(rnn_out)  # (batch_size, width', vocab_size)
        return logits
    
def get_vocab(vocab_file):
    char_list = []
    with open(vocab_file, "r") as f:
        for line in f:
            char_list.append(line[:-1])

    char_list = ['<blank>'] + char_list 

    char_to_idx = {c: i for i, c in enumerate(char_list)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    num_classes = len(char_to_idx)
    blank_idx = 0

    return char_to_idx, idx_to_char, num_classes, blank_idx

def encode(transcription, char_to_idx):
    return [char_to_idx[c] for c in transcription]

def decode(transcription, idx_to_char):
    return "".join([idx_to_char[i] for i in transcription])

def collate_fn(batch):
    images, transcriptions = zip(*batch)
    images = torch.stack(images, 0)  # Stack images into a batch tensor
    transcription_lengths = [len(t) for t in transcriptions]
    transcriptions_padded = nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in transcriptions],
        batch_first=True, padding_value=0  # Pad with blank index
    )
    return images, transcriptions_padded, transcription_lengths

def decode_predictions(logits, idx_to_char):
    """
    Decode predictions using greedy decoding.
    """
    probs = logits.softmax(2)  # Convert logits to probabilities
    preds = torch.argmax(probs, dim=2)  # Get the most likely class for each timestep
    pred_transcriptions = []
    for pred in preds:
        transcription = []
        prev_char = None
        for idx in pred:
            if idx != prev_char and idx != dataset.blank_idx:  # Remove duplicates and blanks
                transcription.append(idx_to_char[idx.item()])
            prev_char = idx
        pred_transcriptions.append("".join(transcription))
    return pred_transcriptions

def calculate_character_accuracy(predictions, ground_truth):
    """
    Calculate character-level accuracy.
    """
    total_chars = 0
    correct_chars = 0
    for pred, gt in zip(predictions, ground_truth):
        total_chars += len(gt)
        correct_chars += sum(1 for p, g in zip(pred, gt) if p == g)
    return correct_chars / total_chars












