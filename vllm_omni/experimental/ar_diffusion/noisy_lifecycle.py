# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which pool key a Noisy PP request uses, and what happens to it afterwards.

The AR-Diffusion runner and the CausVidWan pipeline both need this answer: the
runner allocates, binds and releases the pool, and the pipeline looks the same key
up when it plans its cells and checks residency. They used to derive it
separately, which let a streaming ``tick`` (the runner's authoritative view) and a
plain ``extra_args`` read (the pipeline's view) disagree about the key. One
implementation removes the possibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoisyRequestLifecycle:
    """Identity and lifecycle a Noisy PP request carries.

    ``kv_id`` is the pool key: the outer ``session_id`` for a retained session and
    the request id for an ephemeral one. ``persist`` says whether the versions
    survive the request, ``reset`` releases the key before the forward, and
    ``close_session`` releases it afterwards.
    """

    kv_id: str
    persist: bool
    reset: bool
    close_session: bool


def noisy_request_lifecycle(
    request_id: str,
    sampling_params: object,
    *,
    tick: object | None = None,
    fallback: str = "",
) -> NoisyRequestLifecycle:
    """Derive a request's Noisy PP identity and lifecycle from its own fields.

    A streaming ``tick`` carries the authoritative session state when present;
    otherwise the request's ``extra_args`` decide. An explicit non-empty
    ``session_id`` makes the key persistent (except for an explicit close), and an
    absent one leaves the request id as an ephemeral key.
    """
    if tick is not None:
        close_session = bool(getattr(tick, "close_session", False))
        return NoisyRequestLifecycle(
            kv_id=str(getattr(tick, "session_id", "") or ""),
            persist=not close_session,
            reset=bool(getattr(tick, "reset", False)),
            close_session=close_session,
        )
    extra = getattr(sampling_params, "extra_args", None)
    extra_args = extra if isinstance(extra, dict) else {}
    raw = extra_args.get("session_id")
    explicit = raw is not None and str(raw) != ""
    close_session = bool(extra_args.get("close_session", False))
    return NoisyRequestLifecycle(
        kv_id=str(raw) if explicit else str(request_id or fallback or ""),
        persist=explicit and not close_session,
        reset=bool(extra_args.get("reset", False)),
        close_session=close_session,
    )
