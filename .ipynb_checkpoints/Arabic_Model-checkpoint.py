import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")



df = pd.read_excel('dataset.xlsx')
texts = df['content'].tolist()

tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-large-arabic')
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

model = AutoModel.from_pretrained('asafaya/bert-large-arabic')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}

batch_size = 16
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

learning_rate = 2e-5
weight_decay = 0.002
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

epochs = 300
total_steps = len(dataloader) * epochs
warmup_steps = int(0.15 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

loss_values = []
early_stopping_patience = 3
best_loss = float('inf')
no_improvement_epochs = 0

for epoch in range(epochs):
    epoch_loss = 0
    model.train()

    for i, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}

        optimizer.zero_grad()
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        loss = embeddings.norm()

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    loss_values.append(avg_epoch_loss)

    print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            break

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()