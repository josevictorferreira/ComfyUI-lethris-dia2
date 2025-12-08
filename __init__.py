# Lethris dia2 Nodes Init

from .dia2_node import Dia2_TTS_Generator
from .dia2_captions_node import Dia2_Captions_Generator

NODE_CLASS_MAPPINGS = {
    "Dia2_TTS_Generator": Dia2_TTS_Generator,
    "Dia2_Captions_Generator": Dia2_Captions_Generator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dia2_TTS_Generator": "🗣️ Dia2 TTS Generator",
    "Dia2_Captions_Generator": "💬 Dia2 Captions Generator",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]