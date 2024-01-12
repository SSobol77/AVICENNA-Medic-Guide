import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming Encoder, Decoder, CustomDataset, and Tokenizer are already defined
from models.encoder import Encoder
from models.decoder import Decoder
from dataset import CustomDataset, create_data_loader

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model parameters
input_dim = 1024  # Size of the source vocabulary
output_dim = 1024  # Size of the target vocabulary
embed_size = 256  # Size of embedding vectors
num_layers = 3  # Number of layers in the transformer
heads = 8  # Number of attention heads
forward_expansion = 4
dropout = 0.1
max_length = 100  # Maximum length of sequences
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Load the dataset
tokenizer = Tokenizer()
train_dataset = CustomDataset([...], tokenizer, max_length)
train_loader = create_data_loader(train_dataset, batch_size)

# Define the model, loss function, and optimizer
encoder = Encoder(input_dim, embed_size, num_layers, heads, dropout, forward_expansion, device, max_length).to(device)
decoder = Decoder(output_dim, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length).to(device)
model = Transformer(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx["<pad>"])

# Define the training function
def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(loader):
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        
        # Reshape for calculating loss
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # Calculate and backpropagate the loss
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Training loop
clip = 1
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, clip)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

# Save the model
torch.save(model.state_dict(), "transformer_model.pt")
