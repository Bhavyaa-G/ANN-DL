import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
MAX_LENGTH = 100

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        for line in text:
            encodings_dict = tokenizer(line, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun sets in the west and rises in the east.",
    "Artificial Intelligence is transforming the world.",
    "Deep learning models are revolutionizing various industries.",
    "Natural Language Processing is a key area of artificial intelligence.",
    "Machine learning models are data-driven and improve over time.",
    "The future of technology lies in autonomous systems and robotics.",
    "Cloud computing has become the backbone of modern infrastructure."
]
dataset = TextDataset(text_data, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

for epoch in range(EPOCHS):
    for batch in dataloader:
        input_ids, attn_masks = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

model.eval()
prompt = "Artificial Intelligence"
encoded_input = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
generated_ids = model.generate(encoded_input['input_ids'], max_length=MAX_LENGTH, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Generated Text:\n", generated_text)

