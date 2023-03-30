import torch.nn as nn
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


class CnClipWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ChineseCLIPModel.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')
        processor = ChineseCLIPProcessor.from_pretrained('OFA-Sys/chinese-clip-vit-large-patch14')
        self.tokenizer = processor.tokenizer
        self.transform = processor.image_processor

    def get_image_features(self, **kwargs):
        return self.model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        return self.model.get_text_features(**kwargs)