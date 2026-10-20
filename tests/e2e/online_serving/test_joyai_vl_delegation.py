# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real JoyAI delegation over the shared Realtime client.

The unit and session-harness tests prove the plugin side. This module drives the
whole chain on a served deployment: JoyAI chooses to delegate, the session
publishes a standard function call, the application answers it with a
``function_call_output``, and the next observation speaks that answer.
"""

import asyncio
import base64
import io
import json

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import delegation_tool_definition

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni]

#: The answer the application fabricates for the delegated question.
DELEGATED_ANSWER = "the code is ZB-4471"
DELEGATION_INSTRUCTION = (
    "You are watching a camera feed. Speak only when you have something to report. "
    "If a question needs information the image cannot show, delegate it to the background helper "
    "instead of guessing."
)
DELEGATION_PROMPT = "Which internal product code applies to the object in view? Delegate if you cannot read it."


def frame_base64(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def delegation_session_config() -> SessionConfig:
    """A JoyAI session whose only declared tool is the delegation function.

    ``tools`` is a top-level Realtime session field; the engine maps it onto the
    internal ``extra_body["realtime_tools"]`` the plugin validates.
    """
    return SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions=DELEGATION_INSTRUCTION,
        extra_body={"tts_language": "English"},
    )


async def declare_delegation_tool(client: DuplexClient) -> None:
    await client.send({"type": "session.update", "session": {"tools": [delegation_tool_definition()]}})


def function_calls(collector: EventCollector) -> list[dict[str, object]]:
    return [event for event in collector.events if event.get("type") == "response.function_call_arguments.done"]


async def wait_for_function_call(collector: EventCollector, timeout_s: float = 180) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        calls = function_calls(collector)
        if calls:
            return calls[0]
        await asyncio.sleep(0.05)
    raise AssertionError(f"JoyAI published no function call; recent events: {collector.events[-5:]}")


async def observe(client: DuplexClient, color: str):
    await client.append_audio(b"", video_frames=[frame_base64(color)])
    await client.commit(create_response=True)
    return await client.wait_for("response.done", timeout_s=180)


async def run_delegation(url, model, output_dir):
    config = delegation_session_config()
    collector = EventCollector()
    async with DuplexClient(url, model=model, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        try:
            await declare_delegation_tool(client)
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": DELEGATION_PROMPT}],
                    },
                }
            )

            # 1. The model delegates: a declared tool must never surface as an
            #    unsupported call, and the call must carry its own question.
            await observe(client, "red")
            call = await wait_for_function_call(collector)
            arguments = json.loads(str(call.get("arguments") or "{}"))
            assert arguments.get("question"), f"the call must carry a delegated question: {arguments}"
            assert "unsupported_tool_call" not in json.dumps(collector.errors()), collector.errors()

            # 2. The application answers the call asynchronously.
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": DELEGATED_ANSWER,
                    },
                }
            )
            assert not any(error.get("code") == "invalid_function_call_output" for error in collector.errors()), (
                collector.errors()
            )

            # 3. The next observation speaks the delegated answer.
            call_id = str(call["call_id"])
            transcript = ""
            for color in ("red", "blue", "red"):
                terminal = await observe(client, color)
                transcript = collector.response_text(terminal.response_id)
                if DELEGATED_ANSWER.split()[-1].lower() in transcript.lower():
                    break
            (output_dir / "delegation-events.json").write_text(json.dumps(collector.events, indent=2), encoding="utf-8")
            assert DELEGATED_ANSWER.split()[-1].lower() in transcript.lower(), (
                f"the delegated answer never reached speech: {transcript!r}"
            )

            # A repeated result for the accepted call is refused by the ledger.
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": "a second answer"},
                }
            )
            await asyncio.sleep(0.5)
            duplicates = [error for error in collector.errors() if error.get("code") == "invalid_function_call_output"]
            assert duplicates, "a duplicate function_call_output must be refused"

            await client.close()
            await consumer
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model="jdopensource/JoyAI-VL-Interaction-Preview",
            stage_config_path=modify_stage_config(
                get_deploy_config_path("joyai_vl_interaction.yaml"),
                updates={"session_mode": "duplex"},
            ),
        )
    ],
    indirect=True,
)
def test_delegated_question_round_trips_into_speech(omni_server, tmp_path):
    asyncio.run(
        run_delegation(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
        )
    )
