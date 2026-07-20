"""
patched_modeling_stage2.py — DUAL-TRACK modified copy of  src/modeling/modeling_stage2.py
==========================================================================================

Changes vs. the original LatentSFTStage2SoftEmbedding:

__init__
  * attn_implementation is now 'flash_attention_2' if use_flash_attention_2 else 'SDPA'
    (SDPA is required because we feed a custom 4-D additive attention mask; FA2 does
     not accept arbitrary 4-D masks).  Run with --use_flash_attention_2 False.
  * <cot>/</cot> registered as additional special tokens, embeddings resized, the two
    new rows initialised from <think>/</think>, and (embed_tokens, lm_head) added to the
    LoRA ``modules_to_save`` so the new-token rows are actually trained.
  * new loss weights cot_w / ans_w (ce_w/kl_w kept for compatibility).
  * dist.get_rank() calls made safe for single-process (non-DeepSpeed) plain-Trainer runs.

forward
  * accepts a 4-D BOTTLENECK attention mask (bool keep-mask [B,1,S,S] from the collator);
    converts it to an additive float mask in the compute dtype and passes it straight to
    self.latent_model(inputs_embeds=..., attention_mask=<4D>).
  * the single CE is split into loss_cot (labels_cot) + loss_ans (labels_answer);
    loss = cot_w*loss_cot + ans_w*loss_ans + kl_w*loss_kl.  loss_kl is UNCHANGED.

inference
  * new staticmethod dualtrack_generate_hf(...) does two/three-phase decoding with a KV
    cache and, crucially, applies the bottleneck at generation time: the answer-generation
    steps (starting from the </cot> query) are given a 2-D key mask that ZEROES the
    <cot>..</cot> key positions, so the answer cannot attend to the visible CoT KV.
    This is what makes the "change visible CoT -> answer unchanged" verify test decisive.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)

COT_OPEN = "<cot>"
COT_CLOSE = "</cot>"


def _rank0() -> bool:
    """True on rank 0, and always True when torch.distributed is not initialised
    (plain single-process Trainer run without DeepSpeed/torchrun)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


@dataclass
class LatentOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None


def kd_kl_loss(logits, target_probs, mask=None, temperature=1.0, eps=1e-12):
    z = logits.float()
    if temperature != 1.0:
        z = z / temperature
    logp = F.log_softmax(z, dim=-1)
    q = target_probs.float()
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    kl_per_tok = (q * (q.clamp_min(eps).log() - logp)).sum(dim=-1)
    if mask is not None:
        m = mask.to(kl_per_tok.dtype)
        loss = (kl_per_tok * m).sum() / m.sum().clamp_min(1)
    else:
        loss = kl_per_tok.mean()
    return loss * (temperature ** 2)


def kd_kl_loss_topk(student_logits, teacher_probs, teacher_indices, temperature=1.0, eps=1e-12):
    z = student_logits.float()
    if temperature != 1.0:
        z = z / temperature
    student_log_probs = F.log_softmax(z, dim=-1)
    student_log_probs_selected = torch.gather(student_log_probs, -1, teacher_indices)
    q = teacher_probs.float()
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    entropy_term = q * q.clamp_min(eps).log()
    cross_term = q * student_log_probs_selected
    kl_per_token = (entropy_term - cross_term).sum(dim=-1)
    loss = kl_per_token.mean()
    return loss * (temperature ** 2)


def weighted_embedding_from_topk(topk_probs, topk_indices, embedding):
    W = embedding.weight
    topk_indices = topk_indices.to(W.device)
    topk_probs = topk_probs.to(dtype=W.dtype, device=W.device)
    topk_emb = F.embedding(topk_indices, W)
    weighted_emb = topk_emb * topk_probs.unsqueeze(-1)
    return weighted_emb.sum(dim=-2)


def softmax_over_embedding_topk(x, embedding, top_k=50, temperature=1.0, use_cosine=False, eps=1e-12):
    W = embedding.weight.detach()
    x = x.to(dtype=W.dtype, device=W.device)
    if use_cosine:
        x_n = F.normalize(x, p=2, dim=-1, eps=eps)
        W_n = F.normalize(W, p=2, dim=-1, eps=eps)
        logits = F.linear(x_n, W_n)
    else:
        logits = F.linear(x, W)
    if top_k < logits.size(-1):
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    else:
        topk_logits, topk_indices = logits, None
    if temperature != 1.0:
        topk_logits = topk_logits / temperature
    topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(W.dtype)
    if topk_indices is not None:
        topk_emb = F.embedding(topk_indices, W)
        new_x = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1)
        return new_x, topk_probs, topk_indices
    new_x = topk_probs @ W
    return new_x, topk_probs, None


class LatentSFTStage2SoftEmbedding(nn.Module):
    def __init__(self,
        latent_model_path: str = None,
        ce_w: float = 1.,
        kl_w: float = 1.,
        cot_w: float = 1.,          # DUAL-TRACK: visible-CoT CE weight
        ans_w: float = 1.,          # DUAL-TRACK: answer CE weight
        bfloat16: bool = False,
        use_flash_attention_2: bool = False,
        lora_tune: bool = False,
        lora_path: str = None,
        lora_rank: int = 32,
        lora_dropout: float = 0.1,
        save_path: str = None,
        training: bool = True,
        topk_interpolation: int = 10,
        **kwargs
    ):
        super().__init__()
        self.latent_model_path = latent_model_path

        # DUAL-TRACK: SDPA is mandatory for the custom 4-D bottleneck mask.
        attn_impl = 'flash_attention_2' if use_flash_attention_2 else 'sdpa'
        self.latent_model = AutoModelForCausalLM.from_pretrained(self.latent_model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if bfloat16 else torch.float16,
            use_cache=False,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.latent_model_path)
        self.latent_tokens = ['<think>', '</think>']

        if training:
            if self.tokenizer.pad_token is None:
                if 'deepseek' in latent_model_path.lower() or 'qwen' in latent_model_path.lower():
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif 'llama' in latent_model_path.lower():
                    self.tokenizer.pad_token_id = 128001
                else:
                    raise ValueError("Unsupported model type")

        self.latent_token_ids = self.tokenizer(self.latent_tokens, add_special_tokens=False)['input_ids']

        # ------------------------------------------------------------------ #
        #  DUAL-TRACK: register <cot>/</cot> and grow the embedding matrix
        # ------------------------------------------------------------------ #
        self.cot_tokens = [COT_OPEN, COT_CLOSE]
        n_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [COT_OPEN, COT_CLOSE]}
        )
        if n_added > 0:
            self.latent_model.resize_token_embeddings(len(self.tokenizer))
        self.cot_token_ids = self.tokenizer(self.cot_tokens, add_special_tokens=False)['input_ids']

        # Initialise the new rows from <think>/</think> so they start out sensible.
        with torch.no_grad():
            emb = self.latent_model.get_input_embeddings()
            out = self.latent_model.get_output_embeddings()
            src_open = self.latent_token_ids[0][0]
            src_close = self.latent_token_ids[1][0]
            for dst_ids, src in ((self.cot_token_ids[0], src_open),
                                 (self.cot_token_ids[1], src_close)):
                for d in dst_ids:
                    emb.weight[d] = emb.weight[src].clone()
                    if out is not None and out.weight.data_ptr() != emb.weight.data_ptr():
                        out.weight[d] = out.weight[src].clone()

        if training and save_path is not None and _rank0():
            self.save(os.path.join(save_path, 'base_model'))
            self.tokenizer.save_pretrained(os.path.join(save_path, 'base_model'))

        self.topk_interpolation = topk_interpolation
        self.lora_tune = lora_tune
        self.save_path = save_path

        if lora_tune:
            if lora_path is not None:
                self.latent_model = PeftModel.from_pretrained(self.latent_model, lora_path)
            else:
                self.config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    # DUAL-TRACK: train the (resized) token embeddings + lm_head so the
                    # freshly-added <cot>/</cot> rows are learnable under LoRA.
                    modules_to_save=["embed_tokens", "lm_head"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.latent_model = get_peft_model(self.latent_model, self.config)

        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_w = ce_w
        self.kl_w = kl_w
        self.cot_w = cot_w
        self.ans_w = ans_w

    def gradient_checkpointing_enable(self, **kwargs):
        self.latent_model.config.use_cache = False
        self.latent_model.enable_input_require_grads()
        self.latent_model.gradient_checkpointing_enable(**kwargs)

    # ---------------------------------------------------------------------- #
    #  helper: shifted CE over a label tensor
    # ---------------------------------------------------------------------- #
    def _shifted_ce(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.latent_model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        return self.loss_ce(shift_logits, shift_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,   # DUAL-TRACK: 4-D bool keep-mask
        latent_state: list = None,
        latent_index: list = None,
        labels_cot: Optional[torch.LongTensor] = None,     # DUAL-TRACK
        labels_answer: Optional[torch.LongTensor] = None,  # DUAL-TRACK
        labels: Optional[torch.LongTensor] = None,         # legacy (ignored if cot/ans given)
        **kwargs
    ):
        bs = input_ids.shape[0]

        decoder_mask = (input_ids == -100)
        ids_for_embed = input_ids.masked_fill(decoder_mask, self.tokenizer.pad_token_id)

        input_embeddings = self.latent_model.model.model.embed_tokens(ids_for_embed).detach()
        W = self.latent_model.model.model.embed_tokens

        # splice teacher soft latent embeds into the latent region (UNCHANGED)
        rows = []
        for b in range(bs):
            s, e = int(latent_index[b][0]), int(latent_index[b][1])
            x = latent_state[b]
            x = weighted_embedding_from_topk(x[0], x[1], W)
            x = x.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
            assert (e - s) == x.size(0), f"latent length mismatch: {(s, e)} vs {x.shape}"
            row = torch.cat([input_embeddings[b, :s], x, input_embeddings[b, e:]], dim=0)
            rows.append(row)
        input_embeddings = torch.stack(rows, dim=0)

        # DUAL-TRACK: convert 4-D bool keep-mask -> additive float mask (compute dtype)
        attn = attention_mask
        if attn is not None and attn.dim() == 4:
            if attn.dtype == torch.bool:
                add = torch.zeros(attn.shape, dtype=input_embeddings.dtype, device=input_embeddings.device)
                add.masked_fill_(~attn.to(input_embeddings.device), torch.finfo(input_embeddings.dtype).min)
                attn = add
            else:
                attn = attn.to(dtype=input_embeddings.dtype, device=input_embeddings.device)

        decoder_outputs = self.latent_model(inputs_embeds=input_embeddings, attention_mask=attn)
        logits = decoder_outputs.logits

        # ---- DUAL-TRACK: split CE into visible-CoT and answer segments -------
        loss_cot = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        loss_ans = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        if labels_cot is not None:
            loss_cot = self._shifted_ce(logits, labels_cot)
        if labels_answer is not None:
            loss_ans = self._shifted_ce(logits, labels_answer)

        # ---- KL over latent region (UNCHANGED) -------------------------------
        loss_kl = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        for b in range(bs):
            s, e = int(latent_index[b][0]), int(latent_index[b][1])
            assert s >= 1 and e > s, f"bad latent_index: {(s, e)}"
            t_probs, t_indices = latent_state[b]
            t_indices = t_indices.to(device=logits.device, dtype=torch.long)
            t_probs = t_probs.to(device=logits.device, dtype=logits.dtype)
            p_logits = logits[b, s - 1:e - 1]
            loss_kl = loss_kl + kd_kl_loss_topk(p_logits, t_probs, t_indices)
        loss_kl = loss_kl / bs

        loss = self.cot_w * loss_cot + self.ans_w * loss_ans + self.kl_w * loss_kl

        if self.save_path is not None and _rank0():
            with open(os.path.join(self.save_path, 'loss.jsonl'), 'a') as f:
                line = {
                    'loss': float(loss.item()),
                    'loss_cot': float(loss_cot.item()),
                    'loss_ans': float(loss_ans.item()),
                    'loss_kl': float(loss_kl.item()),
                }
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

        return LatentOutput(loss=loss, logits=logits)

    def save(self, output_dir: str):
        state_dict = self.latent_model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.latent_model.save_pretrained(output_dir, state_dict=state_dict)

    # ====================================================================== #
    #  DUAL-TRACK inference with a KV-cache bottleneck
    # ====================================================================== #
    @staticmethod
    @torch.no_grad()
    def dualtrack_generate_hf(
        model,                       # underlying CausalLM: has get_input_embeddings(),
                                     # .latent_token_ids, .cot_token_ids, .tokenizer, .device
        inputs: dict,                # {'input_ids':[1,P]} — prompt ENDING with <bot>(<think>)
        max_latent: int = 64,
        max_cot: int = 256,
        max_ans: int = 64,
        topk_interpolation: int = 10,
        bottleneck: bool = True,
        override_latent_embeds: Optional[torch.Tensor] = None,  # [L,H] donor latent (skip phase A)
        override_cot_ids: Optional[list] = None,                # force visible CoT tokens
    ):
        """Three-phase decode:  latent  ->  <cot> visible-CoT </cot>  ->  answer.

        When ``bottleneck`` is True, every forward pass from the </cot> query onward is
        given a 2-D key mask that zeroes the <cot>..</cot> key positions, so the answer
        is produced from (question + latent + <eot>) only.

        Returns a dict with decoded pieces AND the machinery the verify script needs
        (latent embeds, key spans, token id lists).
        """
        device = model.device
        emb_layer = model.get_input_embeddings()
        W = emb_layer.weight.detach()
        dtype = W.dtype
        eot_id = model.latent_token_ids[1][0]
        cot_open_id = model.cot_token_ids[0][0]
        cot_close_id = model.cot_token_ids[1][0]

        def embed_ids(ids_list):
            t = torch.as_tensor(ids_list, dtype=torch.long, device=device).view(1, -1)
            return emb_layer(t)  # [1,n,H]

        prompt_ids = inputs['input_ids'].to(device)
        P = prompt_ids.shape[1]
        prompt_embeds = emb_layer(prompt_ids)  # [1,P,H]

        kv_len = 0
        past = None

        def step(embeds, key_mask_zero=None):
            """One forward with a growing 2-D attention mask. key_mask_zero = iterable of
            key indices to force to 0 (blocked). Returns last-position logits."""
            nonlocal past, kv_len
            new = embeds.shape[1]
            total = kv_len + new
            attn = torch.ones((1, total), dtype=torch.long, device=device)
            if key_mask_zero:
                for k in key_mask_zero:
                    if 0 <= k < total:
                        attn[0, k] = 0
            out = model(inputs_embeds=embeds, attention_mask=attn,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            kv_len = total
            return out.logits[:, -1, :]

        latent_ids = []
        latent_key_start = P

        # ---------------- Phase A: latent reasoning ----------------
        if override_latent_embeds is not None:
            # prime the prompt, then inject donor latent embeds verbatim
            _ = step(prompt_embeds)
            lat = override_latent_embeds.to(device=device, dtype=dtype)
            if lat.dim() == 2:
                lat = lat.unsqueeze(0)  # [1,L,H]
            logits = None
            for i in range(lat.shape[1]):
                logits = step(lat[:, i:i + 1, :])
            latent_len = lat.shape[1]
            captured_latent = lat.squeeze(0).clone()
        else:
            logits = step(prompt_embeds)
            captured = []
            for _ in range(max_latent):
                next_tok = int(torch.argmax(logits, dim=-1).item())
                if next_tok == eot_id:
                    break
                latent_ids.append(next_tok)
                # soft (top-k interpolated) latent embed fed back, as in the original loop
                topk_logits, topk_idx = torch.topk(logits, k=topk_interpolation, dim=-1)
                topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(dtype)
                topk_emb = F.embedding(topk_idx, W)
                soft = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1).unsqueeze(1)  # [1,1,H]
                captured.append(soft)
                logits = step(soft)
            latent_len = len(captured)
            captured_latent = torch.cat(captured, dim=1).squeeze(0).clone() if captured \
                else torch.empty(0, W.shape[1], device=device, dtype=dtype)

        # ---------------- <eot> ----------------
        eot_pos = kv_len
        logits = step(embed_ids([eot_id]))

        # ---------------- Phase B: visible CoT ----------------
        cot_open_pos = kv_len
        logits = step(embed_ids([cot_open_id]))
        cot_content_ids = []
        if override_cot_ids is not None:
            for cid in override_cot_ids:
                cot_content_ids.append(int(cid))
                logits = step(embed_ids([int(cid)]))
        else:
            for _ in range(max_cot):
                next_tok = int(torch.argmax(logits, dim=-1).item())
                if next_tok == cot_close_id:
                    break
                cot_content_ids.append(next_tok)
                logits = step(embed_ids([next_tok]))

        # key positions the answer must NOT see: [<cot> .. </cot>] inclusive
        ccot_pos = kv_len  # position that </cot> will occupy
        cot_block_keys = list(range(cot_open_pos, ccot_pos + 1)) if bottleneck else None

        # ---------------- </cot> (first answer-producing query, bottlenecked) ----------------
        logits = step(embed_ids([cot_close_id]), key_mask_zero=cot_block_keys)

        # ---------------- Phase C: answer ----------------
        answer_ids = []
        for _ in range(max_ans):
            next_tok = int(torch.argmax(logits, dim=-1).item())
            answer_ids.append(next_tok)
            if next_tok == model.tokenizer.eos_token_id:
                break
            logits = step(embed_ids([next_tok]), key_mask_zero=cot_block_keys)

        tok = model.tokenizer
        return {
            "latent_ids": latent_ids,
            "latent_embeds": captured_latent,          # [L,H]
            "latent_len": latent_len,
            "latent_key_span": [latent_key_start, latent_key_start + latent_len],
            "cot_ids": cot_content_ids,
            "cot_key_span": [cot_open_pos, ccot_pos + 1],
            "answer_ids": answer_ids,
            "latent_text": tok.decode(latent_ids, skip_special_tokens=False) if latent_ids else "",
            "cot_text": tok.decode(cot_content_ids, skip_special_tokens=False),
            "answer_text": tok.decode(answer_ids, skip_special_tokens=True),
            "full_text": (
                "<bot>" + (tok.decode(latent_ids, skip_special_tokens=False) if latent_ids else "")
                + "<eot><cot>" + tok.decode(cot_content_ids, skip_special_tokens=False)
                + "</cot>" + tok.decode(answer_ids, skip_special_tokens=True)
            ),
        }

    # -- original inference helpers kept verbatim for compatibility ---------- #
    @staticmethod
    @torch.no_grad()
    def one_example_generate_hf(model, inputs, max_new_tokens=50, temperature=1.0,
                                top_p=0.9, do_sample=True, topk_interpolation=10,
                                add_gumbel_noise=False, gumbel_temperature=1.0, noise_scale=1.0):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        generated_ids, past_key_values, latent_states = [], None, []
        W = model.get_input_embeddings().weight.detach()
        for _ in range(max_new_tokens):
            if input_ids is not None:
                input_embeddings = model.get_input_embeddings()(input_ids)
            outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask,
                            past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            topk_logits, topk_indices = torch.topk(next_token_logits, k=topk_interpolation, dim=-1)
            topk_probs = torch.softmax(topk_logits.float(), dim=-1).to(W.dtype)
            topk_emb = F.embedding(topk_indices, W)
            input_embeddings = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1),
                                        device=attention_mask.device)], dim=1)
            if next_token[0, 0].item() == model.latent_token_ids[1][0]:
                break
            input_ids = None
            generated_ids.append(int(next_token.item()))
            latent_states.append(input_embeddings)
        return {"text": model.tokenizer.decode(generated_ids, skip_special_tokens=False),
                "generate_token_num": len(generated_ids)}
