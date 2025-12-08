# dia2_captions_node.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

import folder_paths

CATEGORY = "lethris🧠/Dia2"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_unique_path(base_path: Path) -> Path:
    """Return a unique path by appending _001, _002, etc."""
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    i = 1
    while True:
        new_path = parent / f"{stem}_{i:03d}{suffix}"
        if not new_path.exists():
            return new_path
        i += 1


def format_time(t: float, vtt=False, ass=False) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    ms = int((t - int(t)) * 1000)

    if vtt:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"
    if ass:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{ms:02d}"

    # SRT default
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


# ------------------------------------------------------------
# Grouping: Sentence Advanced
# ------------------------------------------------------------
def group_words_advanced(timestamps_json: str) -> list[tuple[str, float, float]]:
    import re
    data = json.loads(timestamps_json)
    captions = []

    buffer_words = []
    start_time = None

    sentence_end_re = re.compile(r'[\.\!\?]$')
    action_end_re = re.compile(r'\)$')

    for word, t in data:
        if start_time is None:
            start_time = t
        buffer_words.append(word)

        if sentence_end_re.search(word) or action_end_re.search(word):
            caption_text = " ".join(buffer_words)
            captions.append((caption_text, start_time, t))
            buffer_words = []
            start_time = None

    if buffer_words:
        captions.append((" ".join(buffer_words), start_time, data[-1][1]))

    return captions


# ------------------------------------------------------------
# Format Converters
# ------------------------------------------------------------
def convert_to_srt(captions):
    lines = []
    for i, (text, start, end) in enumerate(captions, 1):
        lines.append(
            f"{i}\n{format_time(start)} --> {format_time(end)}\n{text}\n"
        )
    return "\n".join(lines)


def make_vtt(captions):
    lines = ["WEBVTT\n"]
    for (text, start, end) in captions:
        lines.append(
            f"{format_time(start, vtt=True)} --> {format_time(end, vtt=True)}\n{text}\n"
        )
    return "\n".join(lines)


def make_ass(captions):
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,3,2,2,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for (text, start, end) in captions:
        events.append(
            f"Dialogue: 0,{format_time(start, ass=True)},"
            f"{format_time(end, ass=True)},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(events)


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------
class Dia2_Captions_Generator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timestamps_json": ("STRING", {"multiline": True, "default": "[]"}),
                "mode": (["Per Word", "Sentence", "Sentence Advanced"], {"default": "Per Word"}),
                "format": (
                    ["SRT", "VTT", "ASS/SSA"],
                    {"default": "SRT"}
                ),
            },
            "optional": {
                "save_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption_text",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, timestamps_json: str, mode: str, format: str, save_output: bool = True):

        data = json.loads(timestamps_json)

        # --------------------------
        # GROUPING MODES
        # --------------------------
        if mode == "Per Word":
            captions = [(w, t, t + 0.5) for w, t in data]

        elif mode == "Sentence":
            captions = []
            buffer_words = []
            start_time = None
            for word, t in data:
                if start_time is None:
                    start_time = t
                buffer_words.append(word)
                if word.endswith(('.', '!', '?')):
                    captions.append((" ".join(buffer_words), start_time, t))
                    buffer_words, start_time = [], None
            if buffer_words:
                captions.append((" ".join(buffer_words), start_time, data[-1][1]))

        else:  # Sentence Advanced
            captions = group_words_advanced(timestamps_json)

        # --------------------------
        # FORMAT CONVERSION
        # --------------------------
        if format == "SRT":
            text = convert_to_srt(captions)
            ext = ".srt"
        elif format == "VTT":
            text = make_vtt(captions)
            ext = ".vtt"
        else:  # ASS/SSA
            text = make_ass(captions)
            ext = ".ass"

        # --------------------------
        # SAVE (optional)
        # --------------------------
        if save_output:
            out_dir = Path(folder_paths.get_output_directory() or "output") / "captions"
            out_dir.mkdir(parents=True, exist_ok=True)
            base_path = out_dir / f"dia2_captions{ext}"
            out_file = _get_unique_path(base_path)

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)

        return (text,)


# Register node
NODE_CLASS_MAPPINGS = {
    "Dia2_Captions_Generator": Dia2_Captions_Generator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dia2_Captions_Generator": "💬 Dia2 Captions",
}
