
import json, random, unicodedata
from pathlib import Path
from config import MODEL_ID, SEED, REFUSAL_PHRASES, MAX_NEW_TOKENS

def seed_all(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def refused(response):
    text = unicodedata.normalize("NFKC", str(response)).casefold().replace("’", "'")
    return any(phrase in text for phrase in REFUSAL_PHRASES)

def save_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def precision():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for model evaluations. Choose A100 in Colab.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)

def load_full_model():
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=precision(), attn_implementation="sdpa").to("cuda").eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

def load_quant_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    cfg=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=precision())
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=cfg, dtype=precision(),
        device_map="auto", attn_implementation="sdpa").eval()

def generate(model, tok, messages, tools=None):
    import torch
    kw = dict(tokenize=True, add_generation_prompt=True,
              enable_thinking=False, return_tensors="pt")
    if tools is not None: kw["tools"] = tools
    x=tok.apply_chat_template(messages, **kw).to(model.device)
    with torch.inference_mode():
        out=model.generate(x, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,x.shape[1]:],skip_special_tokens=True).strip()
