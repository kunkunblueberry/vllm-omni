from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.ming_flash_omni import ming_zimage_transformer, pipeline_ming_imagegen
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import (
    MingZImageTransformer2DModel,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, hidden):
        self.calls.append(hidden.detach().clone())
        return hidden + 10

    @staticmethod
    def zero_negative(value):
        return torch.zeros_like(value)


def _pipeline(monkeypatch):
    pipe = object.__new__(pipeline_ming_imagegen.MingImagePipeline)
    nn.Module.__init__(pipe)
    pipe.register_parameter("_probe", nn.Parameter(torch.zeros(1)))
    pipe.image_gen_config = SimpleNamespace(
        img_gen_scales=[2],
        thinker_hidden_size=3,
        default_height=16,
        default_width=16,
        num_inference_steps=2,
        guidance_scale=2.0,
    )
    pipe.condition_encoder = _ConditionEncoder()
    pipe.byte5 = None
    pipe._dtype = torch.float32
    pipe.device = torch.device("cpu")

    captured = {}

    def fake_forward(_self, z_req):
        captured["request_ids"] = z_req.request_ids
        captured["sampling"] = z_req.sampling_params_list
        captured["positive"] = [x.detach().clone() for x in pipe._pending_prompt_embeds]
        captured["negative"] = [x.detach().clone() for x in pipe._pending_negative_prompt_embeds]
        captured["generator_values"] = [
            torch.rand(1, generator=item.generator).item() for item in z_req.sampling_params_list
        ]
        return DiffusionOutput(output=torch.arange(z_req.num_reqs * 4, dtype=torch.float32).reshape(z_req.num_reqs, 1, 2, 2))

    monkeypatch.setattr(ZImagePipeline, "forward", fake_forward)
    return pipe, captured


def _request(request_id, hidden, *, seed, negative=None, reference=None):
    extra = {"thinker_hidden_states": hidden}
    if negative is not None:
        extra["negative_thinker_hidden_states"] = negative
    if reference is not None:
        extra["reference_image"] = reference
    return OmniDiffusionRequest(
        prompt={"prompt": "", "extra": extra},
        sampling_params=OmniDiffusionSamplingParams(seed=seed, height=16, width=16, num_inference_steps=2),
        request_id=request_id,
    )


def test_ming_pipeline_batches_request_local_conditions_and_preserves_order(monkeypatch):
    pipe, captured = _pipeline(monkeypatch)
    assert pipe.supports_request_batch is True
    req_a = _request("A", torch.full((2, 3), 1.0), seed=111, negative=torch.full((2, 3), 7.0))
    req_b = _request("B", torch.full((2, 3), 2.0), seed=222)

    outputs = pipe.forward(DiffusionRequestBatch([req_a, req_b]))

    assert [item.output.flatten()[0].item() for item in outputs] == [0.0, 4.0]
    assert captured["request_ids"] == ["A", "B"]
    assert captured["positive"][0][0, 0].item() == 11.0
    assert captured["positive"][1][0, 0].item() == 12.0
    assert captured["negative"][0][0, 0].item() == 17.0
    assert torch.count_nonzero(captured["negative"][1]) == 0
    assert [g.initial_seed() for g in (s.generator for s in captured["sampling"])] == [111, 222]


def test_ming_seed_isolation_matches_single_request_execution(monkeypatch):
    pipe, batch_capture = _pipeline(monkeypatch)
    req_a = _request("A", torch.ones((2, 3)), seed=111)
    req_b = _request("B", torch.ones((2, 3)), seed=222)
    batch = DiffusionRequestBatch([req_a, req_b])
    pipe.forward(batch)
    _pipeline_single, single_capture = _pipeline(monkeypatch)
    _pipeline_single.forward(DiffusionRequestBatch([_request("A", torch.ones((2, 3)), seed=111)]))
    assert batch_capture["generator_values"][0] == single_capture["generator_values"][0]
    assert batch_capture["generator_values"][0] != batch_capture["generator_values"][1]


def test_ming_preserves_explicit_request_generators(monkeypatch):
    pipe, capture = _pipeline(monkeypatch)
    generator = torch.Generator().manual_seed(987)
    request = OmniDiffusionRequest(
        prompt={"prompt": "", "extra": {"thinker_hidden_states": torch.ones((2, 3))}},
        sampling_params=OmniDiffusionSamplingParams(
            generator=generator,
            seed=123,
            height=16,
            width=16,
            num_inference_steps=2,
        ),
        request_id="generator-request",
    )

    pipe.forward(DiffusionRequestBatch([request]))

    assert capture["sampling"][0].generator is generator


def test_ming_reference_latents_are_indexed_per_request(monkeypatch):
    captured = {}
    context = SimpleNamespace(ref_latent=torch.tensor([[[[1.0]]], [[[2.0]]]]))
    monkeypatch.setattr(ming_zimage_transformer, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(ming_zimage_transformer, "get_forward_context", lambda: context)

    def fake_parent(_self, x, t, cap_feats, patch_size=2, f_patch_size=1):
        captured["x"] = x
        return x, {}

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", fake_parent)
    transformer = object.__new__(MingZImageTransformer2DModel)
    x = [torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)]
    transformer.forward(x, torch.ones(2), [torch.zeros(1, 1), torch.zeros(1, 1)])

    assert captured["x"][0][0, 1, 0, 0].item() == 1.0
    assert captured["x"][1][0, 1, 0, 0].item() == 2.0


def test_ming_preprocessor_marks_wave_compatibility():
    pre = pipeline_ming_imagegen.get_ming_image_pre_process_func(SimpleNamespace())
    req = _request("A", torch.ones((2, 3)), seed=111)
    processed = pre(req)
    assert processed.batch_compatibility_key[0] == "ming_image"
