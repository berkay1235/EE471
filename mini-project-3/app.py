"""EE 471 Mini Project 3 - backend.

Two endpoints:
  POST /api/chat   {"messages": [{"role": "user", "content": "..."}]} -> {"reply": "..."}
  POST /api/image  {"prompt": "..."}                                  -> {"image": "data:image/png;base64,..."}

Chat uses SmolLM2-360M-Instruct, images use sd-turbo (fast, 1-2 steps).
Both models are loaded lazily on first use and run on MPS (Apple M-series),
CUDA or CPU, whichever is available.
"""
import base64
import io

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CHAT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
IMAGE_MODEL = "stabilityai/sd-turbo"

SYSTEM_PROMPT = (
    "You are a friendly assistant inside an image generation app. "
    "When asked, write short, vivid prompts for text-to-image models. Keep answers brief."
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

app = FastAPI(title="EE471 Mini Project 3")

_chat = {}
_image = {}


def get_chat():
    if not _chat:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _chat["tok"] = AutoTokenizer.from_pretrained(CHAT_MODEL)
        _chat["model"] = AutoModelForCausalLM.from_pretrained(CHAT_MODEL).to(DEVICE)
    return _chat["tok"], _chat["model"]


def get_image_pipe():
    if not _image:
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL, torch_dtype=dtype)
        _image["pipe"] = pipe.to(DEVICE)
    return _image["pipe"]


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ImageRequest(BaseModel):
    prompt: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    tok, model = get_chat()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history += [m.model_dump() for m in req.messages[-10:]]  # keep context short
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"reply": reply.strip()}


@app.post("/api/image")
def image(req: ImageRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    pipe = get_image_pipe()
    # sd-turbo is distilled: 1-4 steps, no classifier-free guidance
    img = pipe(req.prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
