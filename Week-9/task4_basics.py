"""EE 471 - Week 9 - Task 4 (HF LLM course, chapter 2)
Kullanım: python task4_basics.py <görev>
Görevler: loadsave, encode, padding, truncation, special
"""
import os
import sys
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "bert-base-cased"
SAVE_DIR = "saved_bert"


def loadsave():
    model = AutoModel.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Kaydedilen dosyalar:", sorted(os.listdir(SAVE_DIR)))
    reloaded = AutoModel.from_pretrained(SAVE_DIR)
    n_params = sum(p.numel() for p in reloaded.parameters())
    print(f"Diskten tekrar yüklendi: {type(reloaded).__name__}, {n_params:,} parametre")


def encode():
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    text = "Using a Transformer network is simple"
    tokens = tok.tokenize(text)
    ids = tok.convert_tokens_to_ids(tokens)
    print("Metin   :", text)
    print("Tokenlar:", tokens)
    print("ID'ler  :", ids)
    print("Decode  :", tok.decode(ids))


def padding():
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    batch = ["I've been waiting for this course my whole life.", "So have I!"]
    out = tok(batch, padding=True)
    print("pad token:", tok.pad_token, "| id:", tok.pad_token_id)
    for text, ids, mask in zip(batch, out["input_ids"], out["attention_mask"]):
        print(f"\n{text}\n  input_ids     : {ids}\n  attention_mask: {mask}")


def truncation():
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    text = "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century."
    full = tok(text)["input_ids"]
    cut = tok(text, truncation=True, max_length=8)["input_ids"]
    print("Truncation yok  :", len(full), "token")
    print("max_length=8    :", len(cut), "token ->", cut)
    print("Decode          :", tok.decode(cut))


def special():
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    print("Özel tokenlar:", tok.special_tokens_map)
    text = "Hello world"
    with_sp = tok(text)["input_ids"]
    without_sp = tok(text, add_special_tokens=False)["input_ids"]
    print("Özel tokenlarla :", tok.convert_ids_to_tokens(with_sp))
    print("Özel tokensız   :", tok.convert_ids_to_tokens(without_sp))

    n = tok.add_special_tokens({"additional_special_tokens": ["[MYTAG]"]})
    print(f"\n{n} yeni özel token eklendi, yeni sözlük boyutu: {len(tok)}")
    print("Tokenize:", tok.tokenize("Hello [MYTAG] world"))
    model = AutoModel.from_pretrained(CHECKPOINT)
    model.resize_token_embeddings(len(tok))
    print("Model embedding matrisi yeni boyuta uyarlandı:", model.get_input_embeddings().weight.shape)


TASKS = {"loadsave": loadsave, "encode": encode, "padding": padding,
         "truncation": truncation, "special": special}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("Kullanım: python task4_basics.py <görev>\nGörevler:", ", ".join(TASKS))
        sys.exit(1)
    TASKS[sys.argv[1]]()
