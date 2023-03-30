from .altclip.model_wrapper import AltClipWrapper
from .cnclip.model_wrapper import CnClipWrapper
from .taiyi_clip.model_wrapper import TaiyiClipWrapper
from .mclip.model_wrapper import MClipWrapper
from .clip_chinese.model_wrapper import ClipChineseWrapper
from .zhclip.model_wrapper import ZhClipWrapper

def get_model(name):
    model = None
    if name == 'altclip':
        model = AltClipWrapper()
    elif name == 'cnclip':
        model = CnClipWrapper()
    elif name == 'taiyiclip':
        model = TaiyiClipWrapper()
    elif name == 'mclip':
        model = MClipWrapper()
    elif name == 'clip-chinese':
        model = ClipChineseWrapper()
    elif name == 'zhclip':
        model = ZhClipWrapper()
    return model