import torch.nn as nn
from models.zhclip import ZhCLIPProcessor, ZhCLIPModel


class ZhClipWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ZhCLIPModel.from_pretrained('hf_models/zh-clip-vit-roberta-large-patch14-v2/')
        processor = ZhCLIPProcessor.from_pretrained('hf_models/zh-clip-vit-roberta-large-patch14-v2/')
        self.tokenizer = processor.tokenizer
        self.transform = processor.image_processor

    def get_image_features(self, **kwargs):
        return self.model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        return self.model.get_text_features(**kwargs)