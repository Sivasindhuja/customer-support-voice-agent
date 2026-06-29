import os
import sys
import asyncio
import json
import time
from livekit import api

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
from pipecat.services.google.llm import GoogleLLMService

from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)

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

        self.candidate_answers = []

        self.followup_counts = {}

    def get_current_node(self):

        if self.current_index < len(self.questions):
            return self.questions[self.current_index]

        return None

    def advance(self):
        self.current_index += 1

    def store_answer(self, question, answer):

        self.candidate_answers.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    def increment_followup(self, question_id):

        current = self.followup_counts.get(
            question_id,
            0,
        )

        self.followup_counts[question_id] = (
            current + 1
        )

    def get_followup_count(self, question_id):

        return self.followup_counts.get(
            question_id,
            0,
        )


interview = InterviewSession()


def generate_grounding_system_instruction():

    current_q = interview.get_current_node()

    if not current_q:

        return (
            "The interview is complete. "
            "Thank the candidate."
        )

    template = APP_CONFIG["system_instructions"]

    return template.format(
        question_text=current_q["question_en"],
        ideal_answer=current_q.get(
            "ideal_answer",
            "",
        ),
        keywords=", ".join(
            current_q["keywords"]
        ),
    )


async def generate_final_feedback(llm):

    interview_data = json.dumps(
        interview.candidate_answers,
        indent=2,
    )

    feedback_prompt = f"""
You are an expert technical interviewer.

Below is the interview transcript.

INTERVIEW DATA:
{interview_data}

Generate SHORT and NATURAL feedback.

Format:

What went well:
- short points

What can be improved:
- short points

Overall impression:
- 2 short sentences

Keep it under 120 words.
"""

    response = await llm.generate(
        feedback_prompt
    )

    return response.text.strip()


class LatencyMetricsObserver(
    BaseObserver
):

    def __init__(self):

        super().__init__()

        self.user_speech_stopped_time = None

    async def on_push_frame(self, data):

        frame = data.frame

        frame_name = (
            frame.__class__.__name__
        )

        if (
            frame_name
            == "VADUserStoppedSpeakingFrame"
        ):

            self.user_speech_stopped_time = (
                time.get_clock_info(
                    "monotonic"
                ).time
            )

        elif (
            frame_name == "TranscriptionFrame"
            and self.user_speech_stopped_time
        ):

            asr_latency = (
                time.get_clock_info(
                    "monotonic"
                ).time
                - self.user_speech_stopped_time
            )

            print(
                f"[METRICS] "
                f"ASR Latency: "
                f"{asr_latency:.3f}s"
            )

        elif frame_name == "MetricsFrame":

            metrics_data = getattr(
                frame,
                "metrics",
                None,
            )

            if not metrics_data:
                return

            for metric in metrics_data:

                if (
                    "GoogleLLMService"
                    in metric.processor
                ):

                    print(
                        f"[METRICS] "
                        f"LLM TTFT: "
                        f"{metric.value:.3f}s"
                    )

                elif (
                    "DeepgramTTSService"
                    in metric.processor
                ):

                    print(
                        f"[METRICS] "
                        f"TTS TTFB: "
                        f"{metric.value:.3f}s"
                    )


async def main():

   
    required_keys = [
        "DEEPGRAM_API_KEY",
        "GEMINI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]


    for key in required_keys:

        if not os.getenv(key):

            print(
                f"Missing environment variable: "
                f"{key}"
            )

            sys.exit(1)

   
    token = (
        api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_identity("interview-agent")
        .with_name("Interview Agent")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=APP_CONFIG[
                    "interview_room_name"
                ],
            )
        )
        .to_jwt()
    )

    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL"),
        token=token,
        room_name=APP_CONFIG[
            "interview_room_name"
        ],
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )


    stt = DeepgramSTTService(
        api_key=os.getenv(
            "DEEPGRAM_API_KEY"
        ),
        model="nova-2-general",
    )

    llm = GoogleLLMService(
        api_key=os.getenv(
            "GEMINI_API_KEY"
        ),
        settings=GoogleLLMService.Settings(
            model="gemini-2.0-flash",
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv(
            "DEEPGRAM_API_KEY"
        ),
        settings=DeepgramTTSService.Settings(
            voice="aura-helios-en",
        ),
    )

    initial_instruction = (
        generate_grounding_system_instruction()
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    initial_instruction
                ),
            }
        ]
    )

    context_aggregator = (
        LLMContextAggregatorPair(
            context
        )
    )

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

    worker = PipelineWorker(
        pipeline
    )

    metrics_observer = (
        LatencyMetricsObserver()
    )

    worker.add_observer(
        metrics_observer
    )

    runner = WorkerRunner()

    await runner.add_workers(worker)

    async def interview_loop():

        while True:

            current_question = (
                interview.get_current_node()
            )

            if not current_question:

                final_feedback = (
                    await generate_final_feedback(
                        llm
                    )
                )

                print(
                    "\n========== FINAL FEEDBACK =========="
                )

                print(final_feedback)

                with open(
                    "interview_feedback.txt",
                    "w",
                    encoding="utf-8",
                ) as f:

                    f.write(
                        final_feedback
                    )

                await worker.queue_frame(
                    TTSSpeakFrame(
                        final_feedback
                    )
                )

                await asyncio.sleep(4)

                await worker.queue_frame(
                    EndFrame()
                )

                break

            question_text = (
                current_question[
                    "question_en"
                ]
            )

            print(
                f"\n[QUESTION] "
                f"{question_text}"
            )

            await worker.queue_frame(
                TTSSpeakFrame(
                    question_text
                )
            )

            # wait for user response
            await asyncio.sleep(15)

    
            candidate_answer = (
            context.messages[-1]["content"]
            if len(context.messages) > 1
                else ""
            )



            print(
                f"[CANDIDATE] "
                f"{candidate_answer}"
            )

            interview.store_answer(
                question_text,
                candidate_answer,
            )

            question_id = (
                current_question["id"]
            )

            followup_count = (
                interview.get_followup_count(
                    question_id
                )
            )

            if followup_count < 3:

                interview.increment_followup(
                    question_id
                )

            interview.advance()

            next_question = (
                interview.get_current_node()
            )

            if next_question:

               
                new_instruction = (
                    generate_grounding_system_instruction()
                )

                context.messages[0] = {
                    "role": "system",
                    "content": new_instruction,
                }



            await asyncio.sleep(1)

    @transport.event_handler(
        "on_first_participant_joined"
    )
    async def on_first_participant_joined(
        transport,
        participant,
    ):

        print(
            f"[INFO] Participant joined: "
            f"{participant}"
        )

        await asyncio.sleep(1)

        asyncio.create_task(
            interview_loop()
        )

    print(
        "Starting Pipecat Interview Agent..."
    )

    first_question = (
        interview.get_current_node()[
            "question_en"
        ]
    )

    print(
        f"Opening Question: "
        f"{first_question}"
    )

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

