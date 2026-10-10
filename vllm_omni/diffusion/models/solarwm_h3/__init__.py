# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3 Stage 2 world-model support for vLLM-Omni.

Ports SolarWM's Self-Gradient Forcing (SGF) student onto the H3 backbone. The
package currently carries the model-semantics layer (geometry, flow arithmetic,
packed-row layout, row timesteps); the pipeline and its AR-Diffusion binding
are added on top of it.
"""
