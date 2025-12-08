\# ComfyUI-lethris-dia2



🗣️ \*\*Dia2 TTS Generator\*\* \& 💬 \*\*Dia2 Captions Generator\*\* for ComfyUI



---



!\[Dia2 Workflow Example](Examples/Dia2\_TTS\_and\_Caption\_Generators.png)



Generate high-quality text-to-speech and captions inside \*\*ComfyUI\*\* with ease. Supports multiple speakers, punctuation-aware sentence grouping, and multiple caption formats.



---



\## Features



\- 🎙️ Generate TTS audio using Dia2-2B  

\- 👥 Multi-speaker support: `\[S1]`, `\[S2]`  

\- 💬 Generate captions in \*\*SRT\*\*, \*\*SSA/ASS\*\*, and \*\*VTT\*\* formats  

\- 📝 Per-word, sentence, or advanced grouping (respects punctuation and parentheses)  

\- 🧩 Optional voice cloning with example samples (`Voice\_Sample\_S1.wav`, `Voice\_Sample\_S2.wav`)  



---



\## Installation



1\. Download Dia2-2B model \& tokenizer from:  

&nbsp;  https://huggingface.co/nari-labs/Dia2-2B/tree/main

2\. Rename the weights file to:  

&nbsp;  Dia2-2B.safetensors

3\. Place the model and tokenizer files in:  

&nbsp;  ComfyUI/models/Dia2/



---



\## Usage in ComfyUI



1\. Drag in the nodes:

&nbsp;  - 🗣️ \*\*Dia2 TTS Generator\*\* → generates audio and timestamps  

&nbsp;  - 💬 \*\*Dia2 Captions Generator\*\* → converts timestamps to captions

2\. Caption options:

&nbsp;  - \*\*Per Word\*\*

&nbsp;  - \*\*Sentence\*\*

&nbsp;  - \*\*Sentence Advanced\*\*

3\. Caption formats: \*\*SRT\*\*, \*\*SSA/ASS\*\*, \*\*VTT\*\*  

4\. Output folder: `output/captions` (auto-generated, avoids overwriting)  



> Tip: Node colors — TTS node → purple, Captions node → blue  



---



\## Example Workflow



\- Workflow JSON: `Examples/Dia2\_TTS\_and\_Caption\_Generators.json`  

\- Example image: `Examples/Dia2\_TTS\_and\_Caption\_Generators.png`  

\- Voice samples: `Voice/Voice\_Sample\_S1.wav`, `Voice/Voice\_Sample\_S2.wav`



These show how to set up multi-speaker prompts and caption generation.



---



\## Notes



\- Always place your Dia2 model in the `ComfyUI/models/Dia2/` folder for proper usage.  

\- If weights are found in `diffusion\_models`, the node will warn you but can still load them.  



---



\## Credits



Massive thanks to \*\*nari-labs\*\* for an absolutely smashing job on Dia2! 🎉



