from _utils import patch_llama_for_multitoken, generate_multitoken
from datasets import load_dataset
import copy
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
patch_llama_for_multitoken()
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Для обучения устанавливаем n_future_tokens = 10,
# но для вычисления loss используем только первую prediction (индекс 0)

model.config.n_future_tokens = 3

import wandb
wandb.init(project="llama_mtp", config={
    "model_name": "meta-llama/Llama-3.2-1B",
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 2,
    "n_future_tokens": 10,
    "block_size": 128,
})
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Группировка токенов в блоки фиксированной длины (block_size)
block_size = 128
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

class LMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

train_dataset = LMDataset(lm_dataset)
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#############################################
# 5. Определение функции generate_multitoken
#############################################
def generate_multitoken(model, inputs, max_new_tokens, n_future_tokens):
    """
    Генерирует текст с использованием патченной модели.
    Вход inputs – тензор input_ids.
    На каждом шаге вызывается модель, которая возвращает логиты формы
      (batch_size, seq_len, n_future_tokens, vocab_size),
    и выбирается argmax по последней позиции для всех future-токенов.
    """
    generated = inputs  # inputs уже является тензором
    steps = max_new_tokens // n_future_tokens
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            outputs = model(generated)
            # Логиты для последней позиции: (batch_size, n_future_tokens, vocab_size)
            last_logits = outputs[0, -1, :, :]
            predicted_tokens = torch.argmax(last_logits, dim=-1)  # (batch_size, n_future_tokens)
            if predicted_tokens.dim() == 1:
                predicted_tokens = predicted_tokens.unsqueeze(0)
            generated = torch.cat([generated, predicted_tokens], dim=1)
    return generated

#############################################
# 6. Настройка оптимизатора и обучение с предиктом и чекпоинтингом
#############################################
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
global_step = 0
print_every = 300  # каждые 300 шагов выводим информацию, предсказываем через generate_multitoken и сохраняем чекпоинт
output_dir = "./llama_finetuned_pure_torch"

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=None, position_ids=None)
        # Если n_future_tokens > 1, выход logits имеет форму:
        # (batch_size, seq_len, n_future_tokens, vocab_size)
        # Для loss используем только первую prediction (индекс 0)
        if outputs.dim() == 4:
            logits = outputs[:, :, 0, :]
        else:
            logits = outputs
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                             )
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        if global_step % print_every == 0:
            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}")
            
            # Генерация sample-текста через generate_multitoken
            sample_prompt = "A story about a cat:"
            sample_input_ids = tokenizer(sample_prompt, return_tensors="pt")["input_ids"].to(device)
            generated_ids = generate_multitoken(model, sample_input_ids,
                                                max_new_tokens=20,
                                                n_future_tokens=model.config.n_future_tokens)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Step {global_step} prediction via generate_multitoken: {generated_text}")
            
            # Логирование в wandb
            wandb.log({
                "global_step": global_step,
                "loss": loss.item(),
                "sample_prediction": generated_text,
                "examples": wandb.Table(
                    columns=["Text", "Generated Text"],
                    data=[[sample_prompt, generated_text]]
                )
            })
            
            # Сохранение чекпоинта
            checkpoint_path = f"{output_dir}/checkpoint-{global_step}.bin"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "average_loss": avg_loss})

#############################################
# 7. Сохранение финальной модели
#############################################
final_model_path = output_dir + "/pytorch_model_final.bin"
torch.save(model.state_dict(), final_model_path)
tokenizer.save_pretrained(output_dir)
print("Final model saved in:", output_dir)
wandb.finish()