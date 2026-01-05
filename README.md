# ViT and Friends

This repository implements **Vision Transformer (ViT)** **from scratch** in PyTorch.

The repo is meant for learning and is not the most performant, although there are a few instances of better practices. 
The repo uses
- Combined **QKV projection** followed by chunking
- Manual LayerNorm implementation
- Use of **einops** for readable tensor transformations



## Getting Started

```bash
git clone https://github.com/grasgor/scratch-vit.git

pip3 install torch torchvision einops --index-url https://download.pytorch.org/whl/cu130
```

### Run example inference
- `inference.py` - load pretrained weights in custom class for inference
```bash
python ./src/inference.py
```
Output - 
```bash
Downloading image from https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg...
Content type: image/jpeg
Image loaded: (600, 467)

Top-5 predictions:
1. pug: 87.86%
2. Appenzeller Sennenhund: 0.46%
3. Norwegian Elkhound: 0.19%
4. Australian Silky Terrier: 0.18%
5. Griffon Bruxellois: 0.18%
```

## TODO

- [x] Implement ViT-B/16 and load pretrained weights  
- [ ] Training scripts for pretraining a custom ViT  
- [ ] Implement ViT-MAE  
- [ ] Train ViT for the ARC-AGI task  
