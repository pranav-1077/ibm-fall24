"""
Step 5: Fine-tune the BLIP captioning model on the synthetic dataset uploaded to Hugging Face.

Prerequisites:
    pip install datasets transformers torch
    Set HF_TOKEN env var.
"""

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor, get_scheduler

import config

os.environ["HF_TOKEN"] = config.HF_TOKEN


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(
            images=item["image"], text=item["text"], padding="max_length", return_tensors="pt"
        )
        return {k: v.squeeze() for k, v in encoding.items()}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(config.HF_DATASET, split="train")
    # reserve last TEST_SPLIT_SIZE rows for evaluation
    dataset = dataset.select(range(len(dataset) - config.TEST_SPLIT_SIZE))

    processor = BlipProcessor.from_pretrained(config.BASE_BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(config.BASE_BLIP_MODEL).to(device)

    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = config.NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            loss = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}  step {step}  loss {loss.item():.4f}")

    model.push_to_hub(config.FINETUNED_BLIP_MODEL)
    print(f"Model pushed to {config.FINETUNED_BLIP_MODEL}")


if __name__ == "__main__":
    train()
