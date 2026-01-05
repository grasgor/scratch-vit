import torch
import torch.nn as nn
from config import ViTConfig
from torchvision import models, transforms
from vit import ViT

from PIL import Image
import requests
from io import BytesIO
import json


    
def load_torchvision_weights(custom_model: nn.Module, num_classes=None):
    # Load torchvision pretrained ViT-B/16
    tv_model = models.vit_b_16(weights='DEFAULT')
    tv_state_dict = tv_model.state_dict()

    # Map keys from torchvision model to your custom ViT
    mapping = {}
    for k, v in tv_state_dict.items():
        # Patch embeddings (conv projection)
        if k.startswith("conv_proj"):
            new_key = k.replace("conv_proj", "patch_embeddings.conv1")
        # CLS token
        elif k == "class_token":
            new_key = "patch_embeddings.cls_token"
        # Positional embeddings
        elif k == "encoder.pos_embedding":
            new_key = "patch_embeddings.pos_embeddings"
        # Encoder final LayerNorm
        elif k == "encoder.ln.weight":
            new_key = "encoder.final_norm.gamma"
        elif k == "encoder.ln.bias":
            new_key = "encoder.final_norm.beta"
        # Encoder blocks
        elif k.startswith("encoder.layers.encoder_layer_"):
            # Extract block number from encoder_layer_X
            parts = k.split('.')
            block_name = parts[2]  # encoder_layer_0, encoder_layer_1, etc.
            block_idx = block_name.replace("encoder_layer_", "")

            # LayerNorm 1
            if "ln_1.weight" in k:
                new_key = f"encoder.layers.{block_idx}.layer_norm1.gamma"
            elif "ln_1.bias" in k:
                new_key = f"encoder.layers.{block_idx}.layer_norm1.beta"
            # LayerNorm 2
            elif "ln_2.weight" in k:
                new_key = f"encoder.layers.{block_idx}.layer_norm2.gamma"
            elif "ln_2.bias" in k:
                new_key = f"encoder.layers.{block_idx}.layer_norm2.beta"
            # Self-attention (combined QKV)
            elif "self_attention.in_proj_weight" in k:
                new_key = f"encoder.layers.{block_idx}.MHSA.w_qkv.weight"
            elif "self_attention.in_proj_bias" in k:
                new_key = f"encoder.layers.{block_idx}.MHSA.w_qkv.bias"
            # Self-attention output projection
            elif "self_attention.out_proj.weight" in k:
                new_key = f"encoder.layers.{block_idx}.MHSA.proj.weight"
            elif "self_attention.out_proj.bias" in k:
                new_key = f"encoder.layers.{block_idx}.MHSA.proj.bias"
            # MLP layer 1 (mlp.0 is the first linear layer)
            elif "mlp.0.weight" in k:
                new_key = f"encoder.layers.{block_idx}.ffn.fc1.weight"
            elif "mlp.0.bias" in k:
                new_key = f"encoder.layers.{block_idx}.ffn.fc1.bias"
            # MLP layer 2 (mlp.3 is the second linear layer)
            elif "mlp.3.weight" in k:
                new_key = f"encoder.layers.{block_idx}.ffn.fc2.weight"
            elif "mlp.3.bias" in k:
                new_key = f"encoder.layers.{block_idx}.ffn.fc2.bias"
            else:
                continue
        # Classification head
        elif k == "heads.head.weight":
            new_key = "cls_head.weight"
        elif k == "heads.head.bias":
            new_key = "cls_head.bias"
        else:
            continue

        mapping[new_key] = v

    # Load mapped state dict
    missing_keys, unexpected_keys = custom_model.load_state_dict(mapping, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # Handle classification head if num_classes is specified
    if num_classes and num_classes != 1000:
        custom_model.cls_head = nn.Linear(custom_model.cls_head.in_features, num_classes)
        nn.init.trunc_normal_(custom_model.cls_head.weight, std=0.02)
        nn.init.zeros_(custom_model.cls_head.bias)

    return custom_model


if __name__ == "__main__":
    config = ViTConfig()
    model = ViT(config)

    # Load pretrained weights (keep 1000-class head for ImageNet inference)
    model = load_torchvision_weights(model)

    # Load ImageNet class labels
    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        response = requests.get(imagenet_labels_url)
        imagenet_labels = json.loads(response.content)
    except:
        print("Warning: Could not load ImageNet labels, will show indices only")
        imagenet_labels = None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"  # Example image

    print(f"Downloading image from {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    print(f"Content type: {response.headers.get('content-type', 'unknown')}")

    img = Image.open(BytesIO(response.content)).convert("RGB")

    print(f"Image loaded: {img.size}")
    img_tensor = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Move input to device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)  # [1, num_classes]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, 5)

    print("\nTop-5 predictions:")
    for i in range(top_idx.shape[1]):
        idx = top_idx[0, i].item()
        prob = top_prob[0, i].item()
        if imagenet_labels:
            label = imagenet_labels[idx]
            print(f"{i+1}. {label}: {prob*100:.2f}%")
        else:
            print(f"{i+1}. Class {idx}: {prob*100:.2f}%")
