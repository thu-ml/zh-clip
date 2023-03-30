import torch.nn as nn
from multilingual_clip import pt_multilingual_clip
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer


class MClipWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
        self.tokenizer = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
        self.transform = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14').image_processor

    def get_image_features(self, **kwargs):
        return self.clip_model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        embs = self.text_encoder.transformer(**kwargs)[0]
        att = kwargs['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.text_encoder.LinearTransformation(embs)