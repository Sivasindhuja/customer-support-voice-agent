import os
import sys
import asyncio
import json
import time
from dotenv import load_dotenv

from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat.transports.livekit.transport import (
    LiveKitTransport,
    LiveKitParams,
)

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google import GoogleLLMService

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.observers.base_observer import BaseObserver

load_dotenv()

with open("config.json", "r", encoding="utf-8") as f:
    APP_CONFIG = json.load(f)

class InterviewSession:
    def __init__(self, dataset_path="interview_qa.json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.questions = self.dataset["questions"]
        self.current_index = 0

    def get_current_node(self):
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def advance(self):
        self.current_index += 1


interview = InterviewSession()


def generate_grounding_system_instruction():
    current_q = interview.get_current_node()

    if not current_q:
        return (
            "The technical interview is complete. "
            "Thank the candidate and conclude the interview."
        )

    template = APP_CONFIG["system_instructions"]
    return template.format(
        question_text=current_q["question_en"],
        ideal_answer=current_q["ideal_answer"],
        keywords=", ".join(current_q["keywords"])
    )


class LatencyMetricsObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        self.user_speech_stopped_time = None

    async def on_push_frame(self, data):
        frame = data.frame
        frame_name = frame.__class__.__name__

        if frame_name == "VADUserStoppedSpeakingFrame":
            self.user_speech_stopped_time = time.get_clock_info('monotonic').time
        
        elif frame_name == "TranscriptionFrame" and self.user_speech_stopped_time:
            asr_latency = time.get_clock_info('monotonic').time - self.user_speech_stopped_time
            print(f"[METRICS] ASR Processing Latency: {asr_latency:.3f}s")

        elif frame_name == "MetricsFrame":
            metrics_data = getattr(frame, "metrics", None)
            if not metrics_data:
                return
            
            for metric in metrics_data:
                if "GoogleLLMService" in metric.processor:
                        print(f"[METRICS] LLM Time to First Token (TTFT): {metric.value:.3f}s")
                elif "DeepgramTTSService" in metric.processor:
                        print(f"[METRICS] TTS Time to First Byte (TTFB): {metric.value:.3f}s")


async def main():
    required_keys = [
        "DEEPGRAM_API_KEY",
        "GEMINI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_TOKEN",
    ]

    for key in required_keys:
        if not os.getenv(key):
            print(f"Missing environment variable: {key}")
            sys.exit(1)

    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL"),
        token=os.getenv("LIVEKIT_API_TOKEN"),
        room_name=APP_CONFIG["interview_room_name"],
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-general",
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GoogleLLMService.Settings(
            model="gemini-2.5-flash",
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(
            voice="aura-helios-en",
        ),
    )

    initial_instruction = generate_grounding_system_instruction()
    first_question = interview.get_current_node()["question_en"]

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": initial_instruction,
            }
        ]
    )

    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(pipeline)
    metrics_observer = LatencyMetricsObserver()
    await worker.add_observer(metrics_observer)

    runner = WorkerRunner()
    await runner.add_workers(worker)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"[INFO] Participant joined: {participant}")
        await asyncio.sleep(1)
        await worker.queue_frame(TTSSpeakFrame(first_question))

    @context_aggregator.user().on_context_updated
    async def check_graceful_degradation(_, messages):
        try:
            token_info = await llm.count_tokens(context)
            input_tokens = token_info.get("total_tokens", 0) if isinstance(token_info, dict) else token_info
            print(f"[METRICS] Total Transaction Input Tokens: {input_tokens}")
            
            if input_tokens > APP_CONFIG["max_context_token_threshold"]:
                warning_message = "System parameter limits reached. Concluding interview track gracefully."
                print(f"[WARN] Token threshold breached ({input_tokens}). Graceful exit triggered.")
                await worker.queue_frame(TTSSpeakFrame(warning_message))
                await asyncio.sleep(4) 
                await worker.queue_frame(EndFrame())
        except Exception as e:
            print(f"[ERROR] Failed token counting tracking: {e}")

    @context_aggregator.assistant().event_handler("on_context_updated")
    async def on_context_updated(_, messages):
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            return

        text = last_message.get("content", "").lower()
        print(f"[ASSISTANT]: {text}")

        if (
            "let's move to the next topic" in text
            or "next question" in text
        ):
            interview.advance()
            current = interview.get_current_node()

            if current:
                new_instruction = generate_grounding_system_instruction()
                context.set_system_instruction(new_instruction)
                next_question = current["question_en"]
                print(f"[INFO] Moving to question {interview.current_index}")
                await asyncio.sleep(1)
                await worker.queue_frame(TTSSpeakFrame(next_question))
            else:
                await worker.queue_frame(
                    TTSSpeakFrame("The interview is complete. Thank you.")
                )

    print("Starting Pipecat Interview Agent...")
    print(f"Opening Question: {first_question}")

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())