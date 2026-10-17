"""The only module that touches MMaDA.

Loads the base model + LoRA + tokenizer, and does backward passes on masked
states. Upstream `vendor/MMaDA` is never modified; it is used through sys.path
exactly as the upstream repo ships it.
"""
import sys
from pathlib import Path

from .constants import LORA_RANK, LORA_TARGETS, MMADA_SRC, MODEL_PATH, VQ_MODEL_PATH

if MMADA_SRC not in sys.path:
    sys.path.insert(0, MMADA_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from models import MAGVITv2, MMadaModelLM  # noqa: E402
from models.configuration_llada import ActivationCheckpointingStrategy  # noqa: E402
from training.prompting_utils import UniversalPrompting  # noqa: E402

# VQ_MODEL_PATH comes from constants (env-overridable); see the module docstring there.

# MMaDA's t2i layout is NOT [prompt | completion]. It is
#   [<|t2i|> | text | <eos> | <|soi|> | 1024 image tokens | <|eoi|>]
# so "prompt" is everything before the image block and "completion" is the block.
SPECIAL = ("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>")
N_VQ = 1024
CODEBOOK = 8192
PAD_ID = 126093


def setup(seed=42, training=True, device="cuda"):
    """Build (model_with_lora, uni_prompting, mask_token_id)."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, uni = load_tokenizer_and_prompting()
    base = MMadaModelLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK,
                                            lora_dropout=0.0,
                                            target_modules=LORA_TARGETS,
                                            bias="none", task_type=None))
    if training:
        model.get_base_model().model.set_activation_checkpointing(
            ActivationCheckpointingStrategy.fine_grained)
    model = model.to(device).eval()
    assert all(("lora_" in n) or not p.requires_grad
               for n, p in model.named_parameters()), "unexpected trainable param"
    return model, uni, base.config.mask_token_id


def padding_bias(ids, pad_id=PAD_ID):
    am = ids != pad_id
    return (am[:, :, None] & am[:, None, :]).unsqueeze(1)


def state_ids(template, state, offset, mask_id):
    """Splice an image-token state (0..8191, or mask_id) into the t2i template.

    The image block sits at positions [-(N_VQ+1):-1]. Concrete codebook ids are
    shifted by `offset` (= text vocab size) to reach their position in the joint
    vocabulary; mask ids are passed through untouched.
    """
    ids = template.clone()
    s = state.reshape(1, N_VQ)
    assert bool(((s == mask_id) | ((s >= 0) & (s < CODEBOOK))).all()), "bad state ids"
    ids[:, -(N_VQ + 1):-1] = torch.where(s == mask_id, s, s + offset)
    assert torch.equal(ids[:, -(N_VQ + 1):-1] == mask_id, s == mask_id)
    return ids


def logits_at(model, template, state, offset, mask_id):
    ids = state_ids(template, state, offset, mask_id)
    return model(ids, attention_bias=padding_bias(ids)).logits[0, -(N_VQ + 1):-1,
                                                               offset:offset + CODEBOOK].float()


def probs_at(model, template, state, offset, mask_id):
    return logits_at(model, template, state, offset, mask_id).softmax(-1)


def load_tokenizer_and_prompting():
    """Tokenizer + UniversalPrompting, shared by training and generation."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    uni = UniversalPrompting(tok, max_text_len=512, special_tokens=SPECIAL,
                             ignore_id=-100, cond_dropout_prob=0.1,
                             use_reserved_token=True)
    return tok, uni


def load_for_generation(checkpoint, device="cuda"):
    """Load base + optional LoRA adapter + the VQ tokenizer, all in eval mode.

    `checkpoint` is a directory holding `adapter_model.safetensors`, or the string
    "base" for the frozen model. Returns (model, uni, vq). Nothing here is
    trainable -- this path exists for generation and scoring only.
    """
    from peft import PeftModel
    from models import MAGVITv2

    _, uni = load_tokenizer_and_prompting()
    base = MMadaModelLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16)
    if checkpoint and str(checkpoint) != "base":
        if not Path(str(checkpoint)).is_dir():
            raise ValueError(f"checkpoint must be a directory or 'base': {checkpoint}")
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = base
    model = model.to(device).eval()
    vq = MAGVITv2.from_pretrained(VQ_MODEL_PATH).to(device).eval()
    return model, uni, vq
