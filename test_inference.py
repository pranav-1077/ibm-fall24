"""
Step 6: Evaluate the fine-tuned BLIP model against the base model using
        RemoteCLIP cosine similarity on the held-out test split.

Prerequisites:
    pip install datasets transformers torch open_clip_torch huggingface_hub pillow
    git clone https://github.com/ChenDelong1999/RemoteCLIP/
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

import open_clip

import config


def load_blip_models(device):
    ft_processor = BlipProcessor.from_pretrained(config.BASE_BLIP_MODEL)
    ft_model = BlipForConditionalGeneration.from_pretrained(config.FINETUNED_BLIP_MODEL).to(device)

    base_processor = BlipProcessor.from_pretrained(config.BASE_BLIP_MODEL)
    base_model = BlipForConditionalGeneration.from_pretrained(config.BASE_BLIP_MODEL).to(device)

    return base_processor, base_model, ft_processor, ft_model


def load_clip_model(device):
    model_name = config.CLIP_MODEL_NAME
    checkpoint_path = hf_hub_download(
        "chendelong/RemoteCLIP",
        f"RemoteCLIP-{model_name}.pt",
        cache_dir=config.CLIP_CHECKPOINTS_DIR,
    )
    print(f"RemoteCLIP-{model_name} loaded from {checkpoint_path}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    clip_model.load_state_dict(ckpt)
    clip_model = clip_model.eval().to(device)

    return clip_model, preprocess, tokenizer


def caption(processor, model, pil_image, device):
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_processor, base_model, ft_processor, ft_model = load_blip_models(device)
    clip_model, preprocess, tokenizer = load_clip_model(device)

    dataset = load_dataset(config.HF_DATASET, split="train")
    test_set = dataset.select(range(len(dataset) - config.TEST_SPLIT_SIZE, len(dataset)))

    base_total, ft_total = 0.0, 0.0

    for i, example in enumerate(test_set):
        pil_image = example["image"]
        true_text = example["text"]

        base_desc = caption(base_processor, base_model, pil_image, device)
        ft_desc = caption(ft_processor, ft_model, pil_image, device)

        print(f"\nSample {i + 1}")
        print(f"  True:        {true_text}")
        print(f"  Base:        {base_desc}")
        print(f"  Fine-tuned:  {ft_desc}")

        with torch.no_grad():
            img_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            img_emb = clip_model.encode_image(img_tensor)

            base_tokens = torch.tensor(tokenizer([base_desc])).long().to(device)
            base_emb = clip_model.encode_text(base_tokens)

            ft_tokens = torch.tensor(tokenizer([ft_desc])).long().to(device)
            ft_emb = clip_model.encode_text(ft_tokens)

        base_sim = F.cosine_similarity(img_emb, base_emb, dim=-1).item()
        ft_sim = F.cosine_similarity(img_emb, ft_emb, dim=-1).item()

        print(f"  Base cosine sim:       {base_sim:.4f}")
        print(f"  Fine-tuned cosine sim: {ft_sim:.4f}")

        base_total += base_sim
        ft_total += ft_sim

    n = len(test_set)
    print(f"\nAverage cosine similarity — base: {base_total / n:.4f}  |  fine-tuned: {ft_total / n:.4f}")


if __name__ == "__main__":
    evaluate()
