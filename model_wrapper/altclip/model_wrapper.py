import torch.nn as nn
from .modeling_altclip import AltCLIP
from .processing_altclip import AltCLIPProcessor


class AltClipWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AltCLIP.from_pretrained("BAAI/AltCLIP")
        processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
        self.tokenizer = processor.tokenizer
        self.transform = processor.image_processor

    def get_image_features(self, **kwargs):
        return self.model.get_image_features(**kwargs)

    def get_text_features(self, **kwargs):
        return self.model.get_text_features(**kwargs)