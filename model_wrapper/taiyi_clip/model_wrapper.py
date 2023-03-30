import torch.nn as nn
from transformers import BertForSequenceClassification, CLIPModel, BertTokenizer, CLIPProcessor


class TaiyiClipWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese')
        self.tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese')
        self.transform = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14').image_processor

    def get_image_features(self, **kwargs):
        return self.clip_model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        return self.text_encoder(kwargs['input_ids']).logits