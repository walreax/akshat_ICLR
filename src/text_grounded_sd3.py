"""
text_grounded_sd3.py
====================
Reviewer question: "Why should the dominant SAE feature (by mean activation) correspond to
prompt concepts at all? Show text-grounded alternatives (features aligned to concept tokens)
vs your argmax rule on the same human set."

For every prompt of the human-rated set and every (block, timestep) cell the MIRAGE pipeline
already uses, this script regenerates the SAME image (SD3-medium, CPU generator seed 42, 28
steps, CFG 7.0 -- identical to sd_activation_extractor.evaluate_concepts) and picks the SAE
latent whose spatial map defines the "concept heatmap" under several selection rules:

  argmax_mean    the current rule: latent with the highest mean activation over image tokens
                 (replicated exactly; validated against the latent_idx already stored in
                 scratch_pull_599/isoloria/sd3_raw.csv)
  attn_corr      TEXT-GROUNDED: latent whose per-token activation map has the highest Pearson
                 correlation with the image->concept-token joint-attention map of the same
                 block/step (concept tokens = occurrences of the prompt's salient object nouns)
  text_token     TEXT-GROUNDED: latent that is most active on the concept tokens' own hidden
                 states in that block (text stream), restricted to latents alive on the image
  attn_map       no SAE: the concept-token attention map itself is the heatmap (DAAM-style)
  random_active  NULL: a uniformly random latent among those alive on the image
  fixed_generic  NULL: one fixed latent per cell (the latent argmax_mean picks most often over
                 the whole benchmark) -- i.e. a prompt-independent feature

Each rule yields centroid_x/centroid_y/concentration per (block, timestep) in exactly the
format sd3_raw.csv uses, so the unchanged feature code (scratch_pull_599/isoloria/
replicate_mirage_features.py) and the unchanged SA-7 composite can be applied per rule.
The image pass is not altered: the attention tap runs alongside the original JointAttnProcessor2_0
and returns its untouched output.

Usage (on the GPU box):
    python text_grounded_sd3.py --titles-json titles_sd3.json --out-dir tg_sd3 [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sd_activation_extractor as SX  # noqa: E402

RULES = ["argmax_mean", "attn_corr", "text_token", "attn_map", "random_active", "fixed_generic"]
N_CLIP = 77
N_T5 = 256
FIELDS = ["concept_id", "pid", "block", "step_index", "timestep_fraction", "rule", "latent_idx",
          "centroid_x", "centroid_y", "concentration", "corr_attn", "n_concept_pos", "fallback",
          "text_mass"]


# ----------------------------------------------------------------------------------------------
# taps (read-only: the model's outputs are never modified)
# ----------------------------------------------------------------------------------------------
class CondAttnTap:
    """Wraps a joint-attention processor. Output = the original processor's output, untouched.
    At target steps additionally computes the conditional-half image->concept-token attention."""

    def __init__(self, original, name, target_steps, store, state):
        self.original, self.name = original, name
        self.target_steps, self.store, self.state = set(target_steps), store, state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        out = self.original(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask, *args, **kwargs)
        step = self.state["calls"].get(self.name, 0)
        self.state["calls"][self.name] = step + 1
        if step in self.target_steps and encoder_hidden_states is not None:
            self.store[(self.name, step)] = self._tap(attn, hidden_states, encoder_hidden_states)
        return out

    @torch.no_grad()
    def _tap(self, attn, hs, ehs):
        heads = attn.heads
        h, e = hs[1:2], ehs[1:2]                         # batch = [uncond, cond] under CFG
        n_txt = e.shape[1]
        assert n_txt == N_CLIP + N_T5, f"unexpected text length {n_txt}"
        q, k = attn.to_q(h), attn.to_k(h)
        hd = k.shape[-1] // heads
        q = q.view(1, -1, heads, hd).transpose(1, 2)
        k = k.view(1, -1, heads, hd).transpose(1, 2)
        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)
        if getattr(attn, "norm_k", None) is not None:
            k = attn.norm_k(k)
        kt = attn.add_k_proj(e).view(1, -1, heads, hd).transpose(1, 2)
        if getattr(attn, "norm_added_k", None) is not None:
            kt = attn.norm_added_k(kt)
        n_img = q.shape[2]
        kj = torch.cat([k, kt], dim=2)
        scale = hd ** -0.5
        pos = self.state["pos"]                          # LongTensor, text-stream positions
        acc = torch.zeros(n_img, device=q.device, dtype=torch.float32)
        mass = 0.0
        for h0 in range(0, heads, 6):
            s = torch.matmul(q[:, h0:h0 + 6].float(), kj[:, h0:h0 + 6].float().transpose(-1, -2)) * scale
            p = torch.softmax(s, dim=-1)[..., n_img:]    # (1, hc, N_img, N_txt)
            acc += p[0][:, :, pos].mean(-1).sum(0)
            mass += p[0].sum(-1).mean(-1).sum().item()
            del s, p
        return acc / heads, mass / heads                  # (N_img,), mean image->text mass


class TextStreamTap:
    def __init__(self, name, target_steps, store):
        self.name, self.target_steps, self.store, self.count = name, set(target_steps), store, 0

    def __call__(self, module, inputs, output):
        step = self.count
        self.count += 1
        if step in self.target_steps and isinstance(output, tuple):
            cands = [t for t in output if torch.is_tensor(t)]
            txt = min(cands, key=lambda t: t.shape[1])
            self.store[(self.name, step)] = txt[1].detach()   # conditional half: (N_txt, d)


# ----------------------------------------------------------------------------------------------
def concept_positions(pipe, prompt, units):
    """Concept-token positions in SD3's 333-long text stream = [77 CLIP ; 256 T5]."""
    spans = [(s, e) for u in units for (s, e, surf) in u["spans"] if prompt[s:e] == surf]
    pos = set()
    t5 = pipe.tokenizer_3(prompt, padding="max_length", max_length=N_T5, truncation=True,
                          add_special_tokens=True, return_offsets_mapping=True)
    content_t5 = []
    for i, (a, b) in enumerate(t5["offset_mapping"]):
        if b > a:
            content_t5.append(N_CLIP + i)
            if any(a < e and b > s for s, e in spans):
                pos.add(N_CLIP + i)
    cl = pipe.tokenizer(prompt, padding="max_length", max_length=N_CLIP, truncation=True,
                        add_special_tokens=True, return_offsets_mapping=True)
    for i, (a, b) in enumerate(cl["offset_mapping"]):
        if b > a and any(a < e and b > s for s, e in spans):
            pos.add(i)
    fallback = len(pos) == 0
    if fallback:
        pos = set(content_t5)
    return sorted(pos), fallback


def pearson_to_map(codes, a):
    zc = codes - codes.mean(0, keepdim=True)
    ac = a - a.mean()
    den = torch.sqrt((zc ** 2).sum(0) * (ac ** 2).sum()).clamp_min(1e-12)
    return (zc * ac[:, None]).sum(0) / den


def map_stats(heat):
    cx, cy = SX.heatmap_centroid(heat)
    return cx, cy, SX.attention_concentration(heat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imageMetric/sd3_output/sd3_sampled_prompts.csv")
    ap.add_argument("--concepts", default="concept_spans_evalset.json")
    ap.add_argument("--titles-json", required=True, help="JSON list of concept_ids (titles) to run")
    ap.add_argument("--fixed-latents", default="fixed_latents_sd3.json")
    ap.add_argument("--sae-dir", default="./sae_checkpoints")
    ap.add_argument("--blocks", default="4,8,12,16,20")
    ap.add_argument("--timesteps", default="0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--map-size", type=int, default=64)
    ap.add_argument("--out-dir", default="tg_sd3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = json.load(open(args.titles_json))
    concepts = json.load(open(args.concepts))
    fixed = json.load(open(args.fixed_latents))
    data = {d["concept_id"]: d for d in SX.load_concept_dataset(args.dataset)}
    todo = [t for t in wanted if t in data]
    missing = [t for t in wanted if t not in data]
    if missing:
        print(f"[warn] {len(missing)} requested titles not in dataset: {missing[:3]} ...")
    if args.limit:
        todo = todo[:args.limit]

    csv_path = os.path.join(args.out_dir, "tg_metrics.csv")
    done = set()
    if os.path.exists(csv_path):
        import pandas as pd
        prev = pd.read_csv(csv_path, usecols=["concept_id", "block", "timestep_fraction", "rule"])
        need = len(args.blocks.split(",")) * len(args.timesteps.split(",")) * len(RULES)
        cnt = prev.groupby("concept_id").size()
        done = set(cnt[cnt >= need].index)
        print(f"[resume] {len(done)} prompts already complete")
    todo = [t for t in todo if t not in done]
    print(f"[main] {len(todo)} prompts to process")

    # ---- load SD3 WITHOUT the usual enable_model_cpu_offload(): on this shared GPU only ~9 GB is
    # free and fp16 T5-XXL alone is ~9.5 GB. Phase A encodes every prompt once with T5 streamed
    # layer-by-layer (accelerate.cpu_offload: same fp16 kernels => same embeddings as a resident
    # T5), caches the embeddings on disk, then frees all text encoders. Phase B generates from the
    # cached embeddings (pipe.encode_prompt is patched to return them), so images are the same.
    from accelerate import cpu_offload
    from diffusers import StableDiffusion3Pipeline
    dev = torch.device(args.device)
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                    torch_dtype=torch.float16)
    emb_dir = os.path.join(args.out_dir, "embeds")
    os.makedirs(emb_dir, exist_ok=True)

    def emb_path(title):
        return os.path.join(emb_dir, f"{zlib.crc32(title.encode()):08x}.pt")

    need = [t for t in todo if not os.path.exists(emb_path(t))]
    if need:
        print(f"[phase A] encoding {len(need)} prompts (T5 streamed layer-by-layer)")
        cpu_offload(pipe.text_encoder_3, execution_device=dev)
        pipe.text_encoder.to(dev)
        pipe.text_encoder_2.to(dev)
        with torch.no_grad():
            for i, t in enumerate(need, 1):
                outs = pipe.encode_prompt(prompt=data[t]["prompt"], prompt_2=None, prompt_3=None,
                                          do_classifier_free_guidance=True, device=dev,
                                          num_images_per_prompt=1, max_sequence_length=N_T5)
                torch.save([o.detach().cpu() for o in outs], emb_path(t))
                if i % 20 == 0 or i == len(need):
                    print(f"  [phase A] {i}/{len(need)}", flush=True)
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.text_encoder_3 = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    transformer = pipe.transformer
    orig_procs = dict(transformer.attn_processors)
    cache = {}

    def fake_encode_prompt(*a, device=None, **k):
        pe, npe, pp, npp = cache["cur"]
        d = device if device is not None else dev
        return pe.to(d), npe.to(d), pp.to(d), npp.to(d)

    pipe.encode_prompt = fake_encode_prompt

    class Runner:
        """Same step-numbering convention as SD3InternalsExtractor.extract (callback sets step+1)."""
        def __init__(self):
            self.be = SX.TransformerActivationExtractor(transformer)

        def run(self, prompt, steps, guidance, block_names, target_steps, generator):
            be = self.be
            be.clear()
            be.step_stride = 1
            be.target_steps = set(target_steps)
            be.block_names = block_names
            be.attach()

            def cb(pipe_, step_index, timestep, kwargs):
                be.set_step(step_index + 1)
                return kwargs
            try:
                out = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                           generator=generator, callback_on_step_end=cb)
            finally:
                be.detach()
            return {"image": out.images[0], "block_activations": dict(be.activations)}

    runner = Runner()
    blocks = [f"transformer_blocks.{b.strip()}" for b in args.blocks.split(",")]
    fracs = [float(t) for t in args.timesteps.split(",")]
    step_idx = sorted({int(round(f * (args.steps - 1))) for f in fracs})
    frac_by_step = {int(round(f * (args.steps - 1))): f for f in fracs}

    saes = {}
    for b in blocks:
        sd = torch.load(os.path.join(args.sae_dir, f"sae_{b.replace('.', '_')}.pt"), map_location=args.device)
        d_hidden, d_in = sd["encoder.weight"].shape
        sae = SX.SparseAutoencoder(d_in, d_hidden).to(args.device)
        sae.load_state_dict(sd)
        saes[b] = sae.eval()

    new_file = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    t0 = time.time()
    for n, title in enumerate(todo, 1):
        item = data[title]
        prompt = item["prompt"]
        units = concepts.get(title, {}).get("units", [])
        pid = concepts.get(title, {}).get("id", "")
        pos, fallback = concept_positions(pipe, prompt, units)
        pos_t = torch.tensor(pos, device=args.device)

        attn_store, text_store = {}, {}
        state = {"calls": {}, "pos": pos_t}
        procs = dict(orig_procs)
        for b in blocks:
            key = f"{b}.attn.processor"
            procs[key] = CondAttnTap(orig_procs[key], key, step_idx, attn_store, state)
        transformer.set_attn_processor(procs)
        handles = []
        for b in blocks:
            mod = SX.get_by_path(transformer, b)
            handles.append(mod.register_forward_hook(TextStreamTap(b, step_idx, text_store)))
        cache["cur"] = torch.load(emb_path(title), map_location="cpu")
        try:
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            res = runner.run(prompt, args.steps, args.guidance, blocks, step_idx, gen)
        finally:
            for h in handles:
                h.remove()
            transformer.set_attn_processor(dict(orig_procs))

        rows = []
        for b in blocks:
            sae = saes[b]
            for st in step_idx:
                act = res["block_activations"].get(b, {}).get(st)
                a_pack = attn_store.get((f"{b}.attn.processor", st))
                t_act = text_store.get((b, st))
                if act is None or a_pack is None or t_act is None:
                    print(f"  [skip] {title[:30]} {b} step {st}: missing capture "
                          f"(act={act is not None}, attn={a_pack is not None}, text={t_act is not None})")
                    continue
                cond = SX.select_conditional_batch(act, do_cfg=True).to(args.device).float()   # (N_img, d)
                A, text_mass = a_pack
                side = int(round(cond.shape[0] ** 0.5))
                assert side * side == cond.shape[0]
                with torch.no_grad():
                    codes = sae.encode(cond)                                   # (N_img, d_hidden)
                    mean_act = codes.mean(0)
                    alive = (codes.std(0) > 1e-6) & (mean_act > 0)
                    corr = pearson_to_map(codes, A)
                    corr_m = torch.where(alive, corr, torch.full_like(corr, -2.0))
                    j_arg = int(mean_act.argmax())
                    j_corr = int(corr_m.argmax())
                    ct = sae.encode(t_act[pos_t].float()).mean(0)
                    j_txt = int(torch.where(alive, ct, torch.full_like(ct, -1e9)).argmax())
                    rng = np.random.default_rng(zlib.crc32(f"{title}|{b}|{st}".encode()))
                    alive_idx = alive.nonzero().flatten().cpu().numpy()
                    j_rnd = int(rng.choice(alive_idx)) if len(alive_idx) else j_arg
                    j_fix = int(fixed[f"{b}|{frac_by_step[st]:.2f}"])

                def latent_map(j):
                    m = codes[:, j].reshape(1, 1, side, side)
                    m = F.interpolate(m, size=(args.map_size, args.map_size), mode="bilinear", align_corners=False)
                    return m.squeeze().cpu()

                choices = {"argmax_mean": j_arg, "attn_corr": j_corr, "text_token": j_txt,
                           "random_active": j_rnd, "fixed_generic": j_fix}
                for rule in RULES:
                    if rule == "attn_map":
                        heat = F.interpolate(A.reshape(1, 1, side, side), size=(args.map_size, args.map_size),
                                             mode="bilinear", align_corners=False).squeeze().cpu()
                        j, c_attn = -1, 1.0
                    else:
                        j = choices[rule]
                        heat = latent_map(j)
                        c_attn = float(corr[j]) if torch.isfinite(corr[j]) else 0.0
                    cx, cy, conc = map_stats(heat)
                    rows.append(dict(concept_id=title, pid=pid, block=b, step_index=st,
                                     timestep_fraction=frac_by_step[st], rule=rule, latent_idx=j,
                                     centroid_x=cx, centroid_y=cy, concentration=conc, corr_attn=c_attn,
                                     n_concept_pos=len(pos), fallback=int(fallback), text_mass=text_mass))
        for r in rows:
            w.writerow(r)
        fh.flush()
        el = time.time() - t0
        print(f"[{n}/{len(todo)}] {title[:40]!r} pos={len(pos)}{' (fallback)' if fallback else ''} "
              f"rows={len(rows)} elapsed={el/60:.1f}m eta={(el/n)*(len(todo)-n)/60:.1f}m", flush=True)
        torch.cuda.empty_cache()
    fh.close()
    print("done")


if __name__ == "__main__":
    main()
