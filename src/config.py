class ViTConfig:
    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 embedding_dim=768, 
                 num_heads=12, 
                 num_layers=12, 
                 mlp_ratio=4, 
                 dropout_rate=0.1, 
                 eps=1e-6, 
                 num_classes=1000):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_layers
        self.scale_factor = mlp_ratio
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.num_classes = num_classes
