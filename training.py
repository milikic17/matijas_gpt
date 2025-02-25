# Import necessary modules
import torch
import random
from model import TransformerModel
from config import batch_size, max_iters, eval_interval, learning_rate, device, block_size, n_embd, n_head, n_layer, dropout, vocab_size, pad_token, sos_token, eos_token 


# Set a random seed for reproducibility
torch.manual_seed(1337)
random.seed(1337)

with open('dataset1.txt', 'r', encoding='utf-8') as f:
    pairs = f.read().strip().split('\n')

# Split source and target texts from the input file and add SOS and EOS tokens
source_texts, target_texts = [], []
for pair in pairs:
    source, target = pair.split('\t')
    source_texts.append([int(token) for token in source.split(',') if token.strip().isdigit()])
    target_texts.append([int(token) for token in target.split(',') if token.strip().isdigit()])

# Combine source and target texts
combined_texts = list(zip(source_texts, target_texts))

# Shuffle the combined list
random.shuffle(combined_texts)

# Splitting data into training and validation sets
split_index = int(len(combined_texts) * 0.9)  # 90% for training, 10% for validation
train_combined, val_combined = combined_texts[:split_index], combined_texts[split_index:]

# Separate the source and target texts for training and validation sets
train_source, train_target = zip(*train_combined)
val_source, val_target = zip(*val_combined)

# Convert to lists (if necessary)
train_source = list(train_source)
train_target = list(train_target)
val_source = list(val_source)
val_target = list(val_target)

# training data
train_source_tensor = torch.tensor(train_source, dtype=torch.long)
train_target_tensor = torch.tensor(train_target, dtype=torch.long)

# validation data
val_source_tensor = torch.tensor(val_source, dtype=torch.long)
val_target_tensor = torch.tensor(val_target, dtype=torch.long)


# Function to get batch data
def get_batch(source_data, target_data, batch_size, device):
    idx = torch.randint(0, source_data.size(0), (batch_size,))
    source_batch = source_data[idx]
    target_batch = target_data[idx]

    source_mask = (source_batch != pad_token)
    target_mask = (target_batch != pad_token)

    source_batch = source_batch.to(device)
    target_batch = target_batch.to(device)
    source_mask = source_mask.to(device)
    target_mask = target_mask.to(device)

    return source_batch, target_batch, source_mask, target_mask




# Instantiate the Transformer model with necessary hyperparameters and move it to the specified device
model = TransformerModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
model.to(device) 


# Define the optimizer for training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    print(f"iter: {iter}")
    model.train()  # Set the model to training mode
    source_batch, target_batch, _, target_mask = get_batch(train_source_tensor, train_target_tensor, batch_size, device)
    logits, loss = model(source_batch, target_batch, target_batch, target_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0 or iter == max_iters - 1:
        train_loss = loss.item()
        val_loss = model.evaluateLoss(val_source_tensor, val_target_tensor, batch_size, pad_token)
        val_accuracy = model.evaluateAccuracy(val_source_tensor, val_target_tensor, batch_size, pad_token)
        print(f"Iteration {iter}: Training Loss {train_loss}, Validation Loss {val_loss}, Validation Accuracy {val_accuracy}")
        
        input_seq = val_source_tensor[1].unsqueeze(0).to(device)
        model.generate_next_token(input_seq, 1835)




# Save the trained model
torch.save(model.state_dict(), 'model_parameters.pth')
