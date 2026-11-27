# EE 471 – Mini Project 3: Dream Canvas

Web app: text-to-image generation + voice/text chatbot.

- Image: `stabilityai/sd-turbo` (diffusers)
- Chatbot: `HuggingFaceTB/SmolLM2-360M-Instruct` (transformers)
- Speech-to-text: browser Web Speech API (use Chrome or Safari)
- Backend: FastAPI, frontend: single `static/index.html`

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://localhost:8000 (must be localhost for the microphone to work).
The first image/chat request downloads and loads the models, so it is slow once.

Figma design: https://www.figma.com/design/8lCLwXFYgiDZEvlqt2GvNV/Dream-Canvas---EE471-Mini-Project-3?t=qYfT2UrC9NOYuuSj-1
