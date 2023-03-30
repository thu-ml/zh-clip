import torch.nn as nn
from transformers import CLIPProcessor
from .model import BertCLIPModel


class ClipChineseWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        model_name_or_path = 'YeungNLP/clip-vit-bert-chinese-1M'
        self.model = BertCLIPModel.from_pretrained(model_name_or_path)
        CLIPProcessor.tokenizer_class = 'BertTokenizerFast'
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = processor.tokenizer
        self.transform = processor.image_processor

    def get_image_features(self, **kwargs):
        return self.model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        kwargs.pop('token_type_ids')
        return self.model.get_text_features(**kwargs)