"""Constants shared by every module. No torch, no model imports, so the pure-logic
pieces (loss, student, teacher arithmetic) can be unit-tested without a GPU or the
MMaDA tree on sys.path.

Paths come from the environment so the code is not tied to one machine. Defaults
point at the layout used in the paper's experiments; override them in your shell,
in a `.env`, or by exporting before launching.

    MMADA_OPSD_MODEL      weights for MMaDA-8B-MixCoT
    MMADA_OPSD_MMADA_SRC  the upstream MMaDA checkout (this repo's vendor/MMaDA works)
    MMADA_OPSD_VQ         MAGVITv2 tokenizer weights
"""
import os
from pathlib import Path

# MMaDA t2i sequence: [<|t2i|>|text|<eos>|<|soi|>|1024 image tokens|<|eoi|>]
N_VQ = 1024       # image token positions per image
CODEBOOK = 8192   # VQ codebook size (image-side vocabulary)
PAD_ID = 126093

MASK_TOKEN_ID = 126336  # default; the loaded config's mask_token_id is authoritative

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "up_proj", "ff_proj"]
LORA_RANK = 32

_HERE = Path(__file__).resolve().parent.parent

_DEFAULT_MODEL = "/physis/models/MMaDA-8B-MixCoT"
_DEFAULT_VQ = "/physis/models/magvitv2"

# The vendored MMaDA tree sits next to this package and is the default source.
# Weights are not redistributed here: point these at your local copies.
MODEL_PATH = os.environ.get("MMADA_OPSD_MODEL", _DEFAULT_MODEL)
MMADA_SRC = os.environ.get("MMADA_OPSD_MMADA_SRC", str(_HERE / "vendor" / "MMaDA"))
VQ_MODEL_PATH = os.environ.get("MMADA_OPSD_VQ", _DEFAULT_VQ)
