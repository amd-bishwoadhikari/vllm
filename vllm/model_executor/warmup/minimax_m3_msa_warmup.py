# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import os
from typing import TYPE_CHECKING

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _warmup_aiter_triton_gemm(worker: "Worker") -> None:
    csv_path = os.environ.get("AITER_CONFIG_GEMM_BF16")
    device = worker.device
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

        triton_shapes = set()
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) > 10 and row[10] == "triton":
                        triton_shapes.add((int(row[2]), int(row[3]), int(row[4])))

        model = worker.get_model()
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            n, k = lm_head.weight.shape
            max_tokens = min(
                worker.scheduler_config.max_num_batched_tokens,
                32768,
            )
            for m in (1, 32, max_tokens):
                triton_shapes.add((m, n, k))

        # Exercise both divisible and non-divisible M to pre-warm both
        # EVEN_MN=True and EVEN_MN=False constexpr specializations.
        shapes_to_warm = set()
        for m, n, k in triton_shapes:
            shapes_to_warm.add((m, n, k))
            shapes_to_warm.add((m + 1, n, k))

        for m, n, k in sorted(shapes_to_warm):
            inp = torch.empty((m, k), dtype=torch.bfloat16, device=device)
            w = torch.empty((n, k), dtype=torch.bfloat16, device=device)
            gemm_a16w16(inp, w, dtype=torch.bfloat16)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        logger.info("Warmed up %d AITER Triton GEMM shapes.", len(shapes_to_warm))
    except Exception as exc:
        logger.warning("Failed AITER Triton GEMM warmup: %s", exc)


@instrument(span_name="MiniMax M3 MSA warmup")
def minimax_m3_msa_warmup(worker: "Worker") -> None:
    sparse_module = next(
        (
            module
            for module in worker.get_model().modules()
            if module.__class__.__name__ == "MiniMaxM3SparseAttention"
        ),
        None,
    )
    if sparse_module is None:
        return

    is_sm100 = (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )
    is_rocm = current_platform.is_rocm()
    if not (is_sm100 or is_rocm):
        return

    logger.info("Warming up MiniMax M3 MSA kernels.")

    # 1. Prefill runs:
    # Ensure num_tokens > max_num_reqs so that max_seqlen_q > 1, forcing
    # unified_attention to run with ALL_DECODE=False (prefill mode).
    # We test both divisible and non-divisible token counts so Triton/Gluon
    # precompiles both tt.divisibility = 16 and fallback specializations.
    max_reqs = worker.scheduler_config.max_num_seqs
    for n_tokens in (max_reqs * 2, max_reqs * 2 + 1):
        worker.model_runner._dummy_run(
            num_tokens=n_tokens,
            force_attention=True,
            skip_eplb=True,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )

    # 2. Mixed batch run with odd decode count (num_tokens=35 -> 17 decodes)
    # to warm up paged attention decode with unaligned context_lengths pointers.
    worker.model_runner._dummy_run(
        num_tokens=35,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )

    # 3. On ROCm, warm up decode paths for both multi-split and single-split
    if is_rocm:
        # Intermediate concurrency (num_tokens=32, 33):
        # mono skips (>16 tokens), routing to AITER sparse decode (pa_decode_gluon).
        # num_seqs <= 64 triggers multi-split decode (num_splits > 1 -> ONE_SHOT=False).
        # We test both 32 (divisible) and 33 (non-divisible) to compile both alignments.
        for dec_tokens in (32, 33):
            worker.model_runner._dummy_run(
                num_tokens=dec_tokens,
                uniform_decode=True,
                force_attention=True,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

        # Low concurrency (num_tokens=2): warms up mono decode path (serves 1..16 tokens)
        worker.model_runner._dummy_run(
            num_tokens=2,
            uniform_decode=True,
            force_attention=True,
            skip_eplb=True,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )

        # Max concurrency -> single-split decode (num_splits <= 1 -> ONE_SHOT=True)
        max_seqs = min(
            worker.scheduler_config.max_num_seqs,
            worker.scheduler_config.max_num_batched_tokens,
        )
        if max_seqs > 33:
            worker.model_runner._dummy_run(
                num_tokens=max_seqs,
                uniform_decode=True,
                force_attention=True,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

        # 4. Pre-warm AITER Triton GEMM fallback kernels
        _warmup_aiter_triton_gemm(worker)
