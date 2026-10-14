# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm MiniMax-M3 ViT rotary kernels and setup kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def minimax_vl_triton_warmup(runner: "GPUModelRunner", model: torch.nn.Module) -> None:
    """Warm up MiniMax-M3 Vision Tower 3D RoPE setup and mrope kernels."""
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
    from vllm.model_executor.layers.rotary_embedding.mrope_vit_setup import (
        vit_mrope_setup,
    )
    from vllm.models.minimax_m3.common.vision_tower import (
        MiniMaxVLVisionTransformer,
    )

    visual = next(
        (m for m in model.modules() if isinstance(m, MiniMaxVLVisionTransformer)),
        None,
    )
    if visual is None:
        return

    device = runner.device
    if visual.inv_freq_t.device.type != device.type:
        return

    logger.info("Warming up MiniMax-M3 ViT rotary kernels on GPU.")

    # 1. Warm _vit_mrope_setup_kernel on representative grid
    grids = [[1, 16, 16]]
    rotary_cos, rotary_sin = vit_mrope_setup(
        visual.inv_freq_t,
        visual.inv_freq_h,
        visual.inv_freq_w,
        grids,
        visual.spatial_merge_size,
    )

    # 2. Warm _triton_mrope_forward inside visual attention
    # Exercise both divisible (256) and non-divisible (257) token counts so Triton
    # compiles for both divisible (tokens % 16 == 0) and fallback integer modes.
    attn = visual.encoder.layers[0].self_attn
    for num_tokens in (256, 257):
        for dtype in (torch.bfloat16, torch.float32):
            q = torch.empty(
                (num_tokens, attn.num_heads_per_partition * attn.head_dim),
                dtype=dtype,
                device=device,
            )
            k = torch.empty_like(q)
            triton_mrope(
                q,
                k,
                rotary_cos,
                rotary_sin,
                attn.mrope_section,
                attn.head_dim,
                attn.rotary_dim,
                mrope_interleaved=False,
                is_neox_style=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    logger.info("Finished warming up MiniMax-M3 ViT rotary kernels.")
