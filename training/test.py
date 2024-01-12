import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu

# Assuming all necessary modules and CustomDataset are already defined
from models.encoder import Encoder
from models.decoder import Decoder
from models.transformer import Transformer
from dataset import CustomDataset, create_data_loader
from tokenization import Tokenizer

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = "transformer_model.pt"
model = Transformer(...)  # Initialize with the correct parameters
model.load_state_dict(torch.load(model_path))
model.to(device)

# Prepare the test dataset and data loader
test_dataset = CustomDataset([...], ...)  # Load your test data here
test_loader = create_data_loader(test_dataset, batch_size=32)

# Define the tokenizer
tokenizer = Tokenizer()

# Function to evaluate the model
def evaluate(model, loader, tokenizer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            trg = batch["trg"].to(device)

            # Forward pass
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            
            # Calculate loss
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(output, trg, ignore_index=tokenizer.token2idx["<pad>"])
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(output, dim=1)
            non_pad_elements = trg.ne(tokenizer.token2idx["<pad>"])
            correct = preds.eq(trg).logical_and(non_pad_elements).sum()
            total_correct += correct.item()
            total_tokens += non_pad_elements.sum().item()

            # Prepare data for BLEU calculation
            trg = trg.view(-1, trg.size(0))
            preds = preds.view(-1, trg.size(0))
            for i in range(trg.size(0)):
                trg_tokens = [tokenizer.idx2token[idx] for idx in trg[i] if idx not in [tokenizer.token2idx["<pad>"], tokenizer.token2idx["<s>"], tokenizer.token2idx["</s>"]]]
                pred_tokens = [tokenizer.idx2token[idx] for idx in preds[i] if idx not in [tokenizer.token2idx["<pad>"], tokenizer.token2idx["<s>"], tokenizer.token2idx["</s>"]]]
                all_references.append([trg_tokens])
                all_hypotheses.append(pred_tokens)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens
    bleu_score = corpus_bleu(all_references, all_hypotheses)
    return avg_loss, accuracy, bleu_score

# Evaluate the model
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx["<pad>"])
test_loss, test_accuracy, test_bleu_score = evaluate(model, test_loader, tokenizer)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, BLEU Score: {test_bleu_score:.4f}")
