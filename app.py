import os
import sys
import asyncio
import json
from dotenv import load_dotenv

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline

# Pipecat 1.3.0+ Modern Orchestration Primitives
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat.transports.livekit.transport import (
    LiveKitTransport,
    LiveKitParams,
)

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

load_dotenv()


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

    question_text = current_q["question_en"]
    ideal_answer = current_q["ideal_answer"]
    keywords = ", ".join(current_q["keywords"])

    return f"""
You are James, an expert technical mock interviewer.

CURRENT QUESTION:
{question_text}

IDEAL ANSWER:
{ideal_answer}

KEYWORDS:
{keywords}

INSTRUCTIONS:
1. Evaluate the candidate answer strictly using the ideal answer and keywords.
2. If the answer is weak or incomplete, ask exactly ONE follow-up question.
3. If the answer is good enough, say exactly:
"Let's move to the next topic."
4. Keep responses short and conversational.
"""


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
        room_name="mock-interview-room",
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

    # Wrap worker instantiation cleanly within main execution loop
    worker = PipelineWorker(pipeline)
    runner = WorkerRunner()
    await runner.add_workers(worker)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"[INFO] Participant joined: {participant}")
        await asyncio.sleep(1)
        # Target the running worker wrapper instead of the old deprecated task object
        await worker.queue_frame(TTSSpeakFrame(first_question))

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

    # Start the worker runner properly
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())