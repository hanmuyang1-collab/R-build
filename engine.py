"""
Rhododendron — GeoThinkAI MoE architecture (user-facing release)
================================================================
One file, four presets, any sequence length. Pick a size, load, generate.

    from rhododendron import Rhododendron
    model = Rhododendron.from_preset("small")          # ~0.6B, runs on a laptop
    model = Rhododendron.from_pretrained("./hf_out")   # load trained weights
    ids = model.generate(prompt_ids, max_new=100)      # any seq length works

CLI:
    python model.py                                # list all presets + param counts
    python model.py --preset micro --test          # CPU forward/backward self-test
    python model.py --preset micro --generate "Hi" # demo gen (byte tokenizer)
    python model.py --count                        # count params (meta device)

Plumbing: every component is registry+config driven; load_growth() upgrades
a trained checkpoint into a re-plumbed model without losing abilities.
Identity: "rhododendron" by "GeoThinkAI" baked into config + saved card.
"""
import math, os, sys, json, glob, shutil, argparse, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict

# ----------------------------------------------------------------- presets
# Sizes are total/active params. "full" = the defaults below (~20.1B / ~3.4B).
PRESETS = {
    "micro":  dict(vocab_size=512,  d_model=256,  n_layers=4,  n_heads=8,  kv_heads=2,
                   n_experts=8,  top_k=2, d_ff=512,  dense_first=1, max_seq=512),
    "small":  dict(vocab_size=50257, d_model=1024, n_layers=8,  n_heads=8,  kv_heads=2,
                   n_experts=16, top_k=4, d_ff=1408, dense_first=1, max_seq=4096),
    "medium": dict(vocab_size=50257, d_model=2048, n_layers=16, n_heads=16, kv_heads=4,
                   n_experts=32, top_k=4, d_ff=1408, dense_first=1, max_seq=4096),
    "full":   dict(),   # class defaults below = 20.09B total / 3.38B active
}
REGISTRY = {"attn": {}, "router": {}, "moe": {}, "act": {}, "norm": {}}
def register(kind):
    def wrap(cls): REGISTRY[kind][cls.__name__.lower().replace("_","")] = cls; return cls
    return wrap

# ----------------------------------------------------------------- config
@dataclass
class RhododendronConfig:
    model_type: str = "rhododendron"
    organization: str = "GeoThinkAI"
    model_name: str = "rhododendron"
    vocab_size: int = 50257
    d_model: int = 3072
    n_layers: int = 24
    n_heads: int = 24
    kv_heads: int = 8
    n_experts: int = 32
    top_k: int = 4
    d_ff: int = 2816
    n_shared_experts: int = 0          # 1 = DeepSeek-style shared expert
    dense_first: int = 1               # first N layers dense (stability)
    max_seq: int = 4096                # training hint; inference accepts ANY length (RoPE is dynamic)
    router_type: str = "topkaux"       # "topkaux" | "auxfree"
    aux_coef: float = 0.003
    z_coef: float = 0.0
    bias_update_rate: float = 0.001
    capacity: float = 0.0              # 0 = dropless; else cap factor (e.g. 1.25)
    act: str = "swiglu"
    attn: str = "gqa"
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    def to_dict(s): return asdict(s)
    @classmethod
    def from_dict(cls, d): return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    @classmethod
    def from_preset(cls, name="full", **overrides):
        """User entry point: pick a size, tweak anything. e.g.
        RhododendronConfig.from_preset("small", n_experts=32, top_k=4, max_seq=8192)"""
        assert name in PRESETS, f"unknown preset {name}; choose {list(PRESETS)}"
        base = cls().to_dict(); base.update(PRESETS[name]); base.update(overrides)
        return cls.from_dict(base)

# ----------------------------------------------------------------- basics
@register("norm")
class RmsNorm(nn.Module):
    def __init__(s, d, eps): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x): return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps) * s.w

def rope_cos_sin(seq, hd, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos()[None, None].to(dtype), f.sin()[None, None].to(dtype)

def apply_rope(x, cos, sin):                       # x [b,h,t,d]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

# ----------------------------------------------------------------- attention
@register("attn")
class GQA(nn.Module):
    def __init__(s, c: RhododendronConfig):
        super().__init__(); s.c = c; s.hd = c.d_model // c.n_heads
        assert c.n_heads % c.kv_heads == 0, "n_heads must be divisible by kv_heads"
        s.q = nn.Linear(c.d_model, c.n_heads * s.hd, bias=False)
        s.k = nn.Linear(c.d_model, c.kv_heads * s.hd, bias=False)
        s.v = nn.Linear(c.d_model, c.kv_heads * s.hd, bias=False)
        s.o = nn.Linear(c.n_heads * s.hd, c.d_model, bias=False)
    def forward(s, x, cos, sin):
        b, t, _ = x.shape; c = s.c
        q = apply_rope(s.q(x).view(b, t, c.n_heads, s.hd).transpose(1, 2), cos, sin)
        k = apply_rope(s.k(x).view(b, t, c.kv_heads, s.hd).transpose(1, 2), cos, sin)
        v = s.v(x).view(b, t, c.kv_heads, s.hd).transpose(1, 2)
        try:
            from flash_attn import flash_attn_func
            y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True).transpose(1, 2)
        except Exception:
            k = k.repeat_interleave(c.n_heads // c.kv_heads, dim=1)
            v = v.repeat_interleave(c.n_heads // c.kv_heads, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(b, t, -1))

# ----------------------------------------------------------------- experts
@register("act")
class SwiGLU(nn.Module):
    def __init__(s, di, do): super().__init__(); s.w1 = nn.Linear(di, do, bias=False); s.w2 = nn.Linear(di, do, bias=False); s.w3 = nn.Linear(do, di, bias=False)
    def forward(s, x): return s.w3(F.silu(s.w1(x)) * s.w2(x))

@register("act")
class GeGLU(nn.Module):
    def __init__(s, di, do): super().__init__(); s.w1 = nn.Linear(di, do, bias=False); s.w2 = nn.Linear(di, do, bias=False); s.w3 = nn.Linear(do, di, bias=False)
    def forward(s, x): return s.w3(F.gelu(s.w1(x)) * s.w2(x))

# ----------------------------------------------------------------- routers
class _RouterBase(nn.Module):
    def __init__(s, c): super().__init__(); s.c = c; s.gate = nn.Linear(c.d_model, c.n_experts, bias=False)
    def stats(s, probs): return probs.mean(dim=(0, 1))

@register("router")
class TopkAux(_RouterBase):
    def forward(s, x):
        p = s.gate(x).softmax(-1)
        vals, idx = p.topk(s.c.top_k, dim=-1)
        vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-9)
        f = s.stats(p)
        return vals, idx, s.c.n_experts * (f * f).sum() - 1.0

@register("router")
class AuxFree(_RouterBase):
    """Bias-adjusted top-k, NO aux loss; bias tracked from running load stats."""
    def __init__(s, c):
        super().__init__(c)
        s.register_buffer("bias", torch.zeros(c.n_experts), persistent=True)
        s.register_buffer("freq_acc", torch.zeros(c.n_experts), persistent=True)
        s.register_buffer("n_seen", torch.zeros((), dtype=torch.long), persistent=True)
    def forward(s, x):
        p = (s.gate(x) + s.bias).softmax(-1)
        vals, idx = p.topk(s.c.top_k, dim=-1)
        vals = vals / vals.sum(-1, keepdim=True).clamp_min(1e-9)
        with torch.no_grad():
            s.freq_acc += F.one_hot(idx, s.c.n_experts).float().sum((1, 2)) / idx[0].numel(); s.n_seen += 1
        return vals, idx, x.new_zeros(())
    def update_bias(s, rate=None):
        if s.n_seen == 0: return
        rate = s.c.bias_update_rate if rate is None else rate
        load = s.freq_acc / s.n_seen.clamp_min(1)
        s.bias -= rate * load.sign() * (load - load.mean()).abs().clamp_min(1e-4) * s.c.n_experts
        s.freq_acc.zero_(); s.n_seen.zero_()

# ----------------------------------------------------------------- MoE block
@register("moe")
class FineGrainedMoE(nn.Module):
    def __init__(s, c: RhododendronConfig):
        super().__init__(); s.c = c
        act = REGISTRY["act"][c.act]
        s.experts = nn.ModuleList([act(c.d_model, c.d_ff) for _ in range(c.n_experts)])
        s.shared = act(c.d_model, c.d_ff) if c.n_shared_experts else None
        s.router = {"topkaux": TopkAux, "auxfree": AuxFree}[c.router_type](c)
    def forward(s, x):
        c = s.c; b, t, d = x.shape
        vals, idx, aux = s.router(x)
        out = torch.zeros_like(x)
        flat, fi, fv = x.view(-1, d), idx.view(-1, c.top_k), vals.view(-1, c.top_k)
        cap = int(c.capacity * flat.shape[0] * c.top_k / c.n_experts) if c.capacity else 0
        for e in range(c.n_experts):
            tok, pos = (fi == e).nonzero(as_tuple=True)
            if cap and tok.numel() > cap: tok, pos = tok[:cap], pos[:cap]
            if tok.numel():
                out.view(-1, d).index_add_(0, tok, fv[tok, pos, None] * s.experts[e](flat[tok]))
        if s.shared is not None: out = out + s.shared(x)
        return out, aux
    def update_router(s):
        if hasattr(s.router, "update_bias"): s.router.update_bias()

class Block(nn.Module):
    def __init__(s, c, moe=True):
        super().__init__(); s.c = c
        s.ln1 = RmsNorm(c.d_model, c.norm_eps); s.attn = GQA(c)
        s.ln2 = RmsNorm(c.d_model, c.norm_eps)
        s.ff = FineGrainedMoE(c) if moe else SwiGLU(c.d_model, 8192)
    def forward(s, x, cos, sin):
        x = x + s.attn(s.ln1(x), cos, sin)
        if isinstance(s.ff, FineGrainedMoE): y, aux = s.ff(s.ln2(x))
        else: y, aux = s.ff(s.ln2(x)), torch.zeros((), device=x.device)
        return x + y, aux

# ----------------------------------------------------------------- tokenizers
class ByteTokenizer:
    """Zero-dependency tokenizer so demos work with no downloads. Not for training."""
    vocab_size = 259
    def encode(s, text): return [b + 3 for b in text.encode("utf-8", errors="ignore")]
    def decode(s, ids):  return bytes([int(i) - 3 for i in ids if 3 <= int(i) < 259]).decode("utf-8", errors="ignore")

# ----------------------------------------------------------------- model
class Rhododendron(nn.Module):
    def __init__(s, c: RhododendronConfig = None):
        super().__init__(); s.cfg = c = c or RhododendronConfig()
        s.gradient_checkpointing = False
        s.embed = nn.Embedding(c.vocab_size, c.d_model)
        s.head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        s.blocks = nn.ModuleList([Block(c, moe=(i >= c.dense_first)) for i in range(c.n_layers)])
        s.lnf = RmsNorm(c.d_model, c.norm_eps)
        s.apply(s._init)
    @classmethod
    def from_preset(cls, name="full", device=None, dtype=None, **overrides):
        m = cls(RhododendronConfig.from_preset(name, **overrides))
        return m.to(device=device, dtype=dtype) if device else m
    @classmethod
    def from_pretrained(cls, path, device="cpu", dtype=torch.bfloat16):
        cfg = RhododendronConfig.from_dict(json.load(open(os.path.join(path, "config.json"))))
        m = cls(cfg); sd = {}
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if shards:
            from safetensors import safe_open
            for f in shards:
                with safe_open(f, framework="pt") as h:
                    for k in h.keys(): sd[k] = h.get_tensor(k)
        elif os.path.exists(os.path.join(path, "pytorch_model.bin")):
            sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        else: raise FileNotFoundError(f"no safetensors/pytorch_model.bin in {path}")
        cls.load_growth(m, sd)
        return m.to(device=device, dtype=dtype)
    def _init(s, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    @property
    def lm_head(s): return s.head if s.head is not None else s.embed
    def forward(s, ids, labels=None):
        b, t = ids.shape; c = s.cfg
        cos, sin = rope_cos_sin(t, c.d_model // c.n_heads, c.rope_theta, ids.device, s.embed.weight.dtype)
        x = s.embed(ids); aux = ids.new_zeros((), dtype=torch.float32)
        for blk in s.blocks:
            if s.gradient_checkpointing and s.training:
                x, a = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else: x, a = blk(x, cos, sin)
            aux = aux + a
        logits = F.linear(s.lnf(x), s.lm_head.weight)
        out = {"aux": aux, "logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.float().view(-1, c.vocab_size), labels.view(-1))
        return out
    @torch.no_grad()
    def generate(s, ids, max_new=50, temperature=0.8, top_k=50, eos_ids=()):
        """ids: LongTensor [1, t]. Works at ANY context length (dynamic RoPE)."""
        s.eval(); eos_ids = set(eos_ids)
        for _ in range(max_new):
            logits = s(ids)["logits"][:, -1].float()
            if temperature <= 0: nxt = logits.argmax(-1, keepdim=True)
            else:
                lg = logits / temperature
                if top_k and top_k < lg.shape[-1]:
                    lg[lg < lg.topk(top_k).values[:, -1:]] = float("-inf")
                nxt = torch.multinomial(lg.softmax(-1), 1)
            ids = torch.cat([ids, nxt], 1)
            if int(nxt[0, 0]) in eos_ids: break
        return ids
    @staticmethod
    def load_growth(model, sd):
        tgt = model.state_dict(); fixed, remapped = {}, 0
        for k, v in sd.items():
            if k in tgt and tgt[k].shape == v.shape: fixed[k] = v
            elif ".experts." in k:
                try:
                    ei = int(k.split(".experts.")[1].split(".")[0])
                    nk = k.replace(f".experts.{ei}.", f".experts.{ei % model.cfg.n_experts}.")
                    if nk in tgt and tgt[nk].shape == v.shape: fixed[nk] = v; remapped += 1
                except Exception: pass
        missing, _ = model.load_state_dict(fixed, strict=False)
        print(f"[load_growth] kept={len(fixed)} remapped={remapped} new_init={len(missing)}")
        return model
    def count(s):
        tot = sum(p.numel() for p in s.parameters())
        c = s.cfg; emb = c.vocab_size * c.d_model * (1 if c.tie_embeddings else 2)
        hd = c.d_model // c.n_heads
        attn = c.d_model * (c.n_heads + 2 * c.kv_heads) * hd + c.n_heads * hd * c.d_model
        per_dense = attn + 3 * c.d_model * 8192
        n_moe = c.n_layers - c.dense_first
        act = emb + n_moe * (attn + (c.top_k + c.n_shared_experts) * 3 * c.d_model * c.d_ff) + c.dense_first * per_dense
        return tot, act
    def save_pretrained(s, path, tokenizer=None):
        os.makedirs(path, exist_ok=True)
        sd = {k: v.cpu().contiguous() for k, v in s.state_dict().items()}
        try:
            from safetensors.torch import save_file
            shards = {}
            for k, v in sd.items(): shards.setdefault(hash(k) % 4, {})[k] = v
            for i, sh in enumerate(shards.values()): save_file(sh, os.path.join(path, f"model_{i}.safetensors"), metadata={"format": "pt"})
        except ImportError:
            torch.save(sd, os.path.join(path, "pytorch_model.bin"))
        json.dump(s.cfg.to_dict(), open(os.path.join(path, "config.json"), "w"), indent=2)
        json.dump({"bos_token_id": 50254, "eos_token_id": 50255, "pad_token_id": 50256}, open(os.path.join(path, "generation_config.json"), "w"))
        shutil.copy(os.path.abspath(__file__), os.path.join(path, "model.py"))
        open(os.path.join(path, "README.md"), "w").write(
"""---
license: apache-2.0
tags: [moe, geothinkai, from-scratch]
---
# Rhododendron
MoE by **GeoThinkAI** (`model_type: rhododendron`). Fine-grained expert routing,
GQA+RoPE+RMSNorm, config-driven plumbing. `model.py` included alongside weights.
""")
        if tokenizer is not None: tokenizer.save_pretrained(path)
    def update_routers(s): [b.ff.update_router() for b in s.blocks if isinstance(b.ff, FineGrainedMoE)]

# ----------------------------------------------------------------- CLI
def _count(name):
    with torch.device("meta"):
        m = Rhododendron(RhododendronConfig.from_preset(name)); tot, act = m.count()
    print(f"{name:7s}: {tot/1e9:6.2f}B total / {act/1e9:5.2f}B active")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rhododendron — GeoThinkAI MoE")
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--count", action="store_true", help="print params (no RAM, meta device)")
    ap.add_argument("--test", action="store_true", help="CPU forward/backward + save/reload")
    ap.add_argument("--generate", metavar="TEXT", help="demo generation (untrained unless --load)")
    ap.add_argument("--load", metavar="DIR", help="load trained weights via from_pretrained")
    ap.add_argument("--tokens", type=int, default=50)
    a = ap.parse_args()
    if a.count or (not (a.test or a.generate)):
        for p in PRESETS: _count(p)
    if a.test:
        c = RhododendronConfig.from_preset("micro")
        m = Rhododendron(c); ids = torch.randint(0, 512, (2, 64))
        o = m(ids, labels=torch.roll(ids, -1, 1)); (o["loss"] + 0.003 * o["aux"]).backward()
        print(f"fwd/bwd OK loss={o['loss'].item():.3f} aux={o['aux'].item():.4f}")
        m.update_routers(); m.save_pretrained("/tmp/rhodo_test")
        m2 = Rhododendron.from_pretrained("/tmp/rhodo_test", dtype=torch.float32)
        Rhododendron.load_growth(m2, m.state_dict()); print("save + from_pretrained + growth OK")
    if a.generate:
        if a.load:
            m = Rhododendron.from_pretrained(a.load); tok = ByteTokenizer()
        else:
            m = Rhododendron.from_preset("micro"); tok = ByteTokenizer()
            print("(untrained micro + byte tokenizer demo)")
        ids = torch.tensor([tok.encode(a.generate)])
        out = m.generate(ids, max_new=a.tokens, temperature=0.8)
        print("gen:", tok.decode(out[0].tolist()))
