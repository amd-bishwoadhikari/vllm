# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shapes the mono kernels are compiled for: one MiniMax-M3 TP4 rank.

Every value here is checked against the loaded model in ``weights.py``; a
model that does not match is never routed to the mono path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# One CTA per MI355X CU; the kernels rely on every CTA being co-resident.
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64


def _serving_tp() -> int:
    """VLLM_MONO_TP when set, else the TP of the engine being built.

    This module is first imported while the model is built, inside the engine's
    config context, so the TP read here is the one being served; one build then
    serves TP2, TP4 and TP8. Outside a config context (e.g. a standalone kernel
    test) it falls back to 2.
    """
    env = os.environ.get("VLLM_MONO_TP")
    if env:
        return int(env)
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config.parallel_config.tensor_parallel_size if config else 2


TP = _serving_tp()
TOTAL_Q_HEADS = 64
TOTAL_KV_HEADS = 4
TOTAL_INDEX_HEADS = 4
TOTAL_INTER = 3072

LOCAL_Q_HEADS = TOTAL_Q_HEADS // TP
LOCAL_KV_HEADS = max(1, TOTAL_KV_HEADS // TP)
LOCAL_INDEX_HEADS = max(1, TOTAL_INDEX_HEADS // TP)

# Kernel variants, off unless switched on (measured at TP2):
# VLLM_MONO_K1_PREFETCH: K1's GEMV CTAs issue every folded task's weights before the
# x8 poll, not each task's after the poll and the previous task's GEMV.
# VLLM_MONO_SPLIT_SUB: each 16-head subpass of a split partition is its own task (a
# rank with more than 16 q heads, i.e. TP2, runs them on separate CTAs, not one
# after another).
# VLLM_MONO_DN_SKIP: the wide MoE down loop skips the B reads and MFMA of units past
# the step's last distinct expert (wave-uniform branch) instead of masking their
# result.
K1_PREFETCH = os.environ.get("VLLM_MONO_K1_PREFETCH", "0") == "1"
SPLIT_SUB = os.environ.get("VLLM_MONO_SPLIT_SUB", "0") == "1" and LOCAL_Q_HEADS > 16
DN_SKIP = os.environ.get("VLLM_MONO_DN_SKIP", "0") == "1"


def qkv_rows(idx_heads: int) -> int:
    """Rows of a rank's fused projection, q | k | v | index_q | index_k:
    LOCAL_Q_HEADS q heads, LOCAL_KV_HEADS k and v heads, ``idx_heads`` index q
    heads, and 1 replicated index k head."""
    return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + 1 + idx_heads) * HEAD_DIM


O_K = LOCAL_Q_HEADS * HEAD_DIM

N_ROUTED = 128
TOP_K = 4
SHARED_EXPERT = N_ROUTED  # the fused shared expert's id
MOE_SLOTS = TOP_K + 1
INTER = TOTAL_INTER // TP  # expert intermediate per rank

SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
MAX_SPARSE_KEYS = SPARSE_BLOCK * TOPK_BLOCKS
# the longest context served: the score region spans it
MAX_CONTEXT = 1 << 20
MAX_INDEX_BLOCKS = MAX_CONTEXT // SPARSE_BLOCK
PAGE16 = 16
# How many K/V sides one block's page-16 ids span. vLLM keeps both sides of a
# block in one allocation, so a block covers both sides' pages and V is reached
# from the same page id through a cache view offset by one side's pages; a layout
# holding the sides as separate planes spans one. Only the main KV cache pages
# this way: the index cache is one plane of whole blocks.
PAGE16_SIDES = 2
BLOCK_PAGES = PAGE16_SIDES * SPARSE_BLOCK // PAGE16  # page ids a block spans

MAX_TOKENS = 16  # tokens one mono step serves (the MFMA B operand holds 16)
# indexer context parallelism: a rank computes every index q head
MAX_QKV_ROWS = qkv_rows(TP)
# indexer context parallelism serves a step's requests past this many index
# blocks: its selection costs a fixed ~7 us (S = 1: ~43.5 us/layer at any length)
# that the one-head path's O(n^2) ranking passes between 258 and 297 blocks
# (S = 1 and S = 16 alike; logs/cp_threshold*.log)
INDEX_CP_FROM_BLOCKS = 288


@dataclass(frozen=True)
class IndexHeads:
    """The index q heads in a rank's fused projection, a build parameter:
    ``count`` of them (LOCAL_INDEX_HEADS, or all TP in head order under indexer context
    parallelism) and ``own``, the one this rank's selection scores."""

    count: int = LOCAL_INDEX_HEADS
    own: int = 0

    def __post_init__(self):
        assert self.count in (1, TP, LOCAL_INDEX_HEADS) and 0 <= self.own < max(self.count, 1)

    @property
    def rows(self) -> int:
        return qkv_rows(self.count)

    @property
    def iq_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.own) * HEAD_DIM

    @property
    def ik_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.count) * HEAD_DIM


# without indexer context parallelism: the rank's own index q head only
ONE_INDEX_HEAD = IndexHeads()

LAYER_SLOTS = 128  # mailbox epochs: step * LAYER_SLOTS + layer + 1


class MonoUnsupported(Exception):
    """The loaded model or runtime configuration is outside what mono serves."""
