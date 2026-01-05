import torch
import torch.nn as nn
from config import ViTConfig
from encoder import PatchEmbeddings, Encoder


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config.in_channels,
                                                (config.image_size, config.image_size),
                                                config.embedding_dim,
                                                config.patch_size)
        
        self.encoder = Encoder(config)

        self.cls_head = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x):
        x = self.patch_embeddings(x)         # [B, num_patches + 1, D]
        x = self.encoder(x)             # [B, num_patches + 1, D]
        cls_token_final = x[:, 0]       # Only the [CLS] token
        logits = self.cls_head(cls_token_final)  # [B, num_classes]
        return logits
    

if __name__ == "__main__":
    config = ViTConfig()

    model = ViT(config)

    # Create a dummy input: batch_size=2, 3 channels, 224x224
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    logits = model(x)
    print("Forward pass successful.")
    print("Output shape:", logits.shape)  # Expected: [2, 1000]

    # Optional: check total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
