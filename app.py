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

# class InterviewSession:
#     def __init__(self, dataset_path="interview_qa.json"):
#         with open(dataset_path, "r", encoding="utf-8") as f:
#             self.dataset = json.load(f)

#         self.questions = self.dataset["questions"]
#         self.current_index = 0

#     def get_current_node(self):
#         if self.current_index < len(self.questions):
#             return self.questions[self.current_index]
#         return None

#     def advance(self):
#         self.current_index += 1

class InterviewSession:
    def __init__(self, dataset_path="interview_qa.json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.questions = self.dataset["questions"]
        self.current_index = 0

        # NEW: store interview performance
        self.performance_log = []

    def get_current_node(self):
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def advance(self):
        self.current_index += 1

    # NEW: save performance per question
    def add_performance(
        self,
        question,
        candidate_answer,
        evaluation,
        strengths,
        improvements,
        score,
    ):
        self.performance_log.append(
            {
                "question": question,
                "candidate_answer": candidate_answer,
                "evaluation": evaluation,
                "strengths": strengths,
                "improvements": improvements,
                "score": score,
            }
        )


interview = InterviewSession()

async def evaluate_candidate_answer(llm, question_data, candidate_answer):
    """
    Uses Gemini to evaluate candidate answer
    and returns structured feedback JSON.
    """

    evaluation_prompt = f"""
You are an expert technical interviewer.

Evaluate the candidate answer.

QUESTION:
{question_data["question_en"]}

IDEAL ANSWER:
{question_data["ideal_answer"]}

KEYWORDS:
{", ".join(question_data["keywords"])}

CANDIDATE ANSWER:
{candidate_answer}

Return STRICT JSON ONLY in this format:

{{
  "evaluation": "short evaluation",
  "strengths": ["point1", "point2"],
  "improvements": ["point1", "point2"],
  "score": 0-10,
  "move_next": true_or_false
}}

Scoring Rules:
- 9-10 = excellent
- 7-8 = good
- 5-6 = partial
- below 5 = weak

Do NOT include markdown.
Do NOT include explanation outside JSON.
"""

    response = await llm.generate(evaluation_prompt)

    raw_text = response.text.strip()

    # Safety cleanup
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
        return parsed
    except Exception as e:
        print("[ERROR] Failed to parse evaluation JSON:", e)
        print(raw_text)

        return {
            "evaluation": "Evaluation parsing failed.",
            "strengths": [],
            "improvements": [],
            "score": 0,
            "move_next": True,
        }
    
async def generate_final_feedback(llm):
    """
    Generates overall interview summary
    based on all question evaluations.
    """

    performance_json = json.dumps(
        interview.performance_log,
        indent=2,
    )

    final_feedback_prompt = f"""
You are an expert technical interviewer.

Based on the interview performance data below,
generate a professional structured interview report.

INTERVIEW DATA:
{performance_json}

Generate:

1. Overall performance summary
2. Technical strengths
3. Areas needing improvement
4. Communication assessment
5. Suggested learning roadmap
6. Final overall score out of 10

Keep it professional and concise.
"""

    response = await llm.generate(final_feedback_prompt)

    return response.text


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

        if last_message.get("role") != "user":
            return

        candidate_answer = last_message.get("content", "").strip()

        current_question = interview.get_current_node()

        if not current_question:
            return

        print(f"[CANDIDATE]: {candidate_answer}")

        # Evaluate answer
        evaluation = await evaluate_candidate_answer(
            llm,
            current_question,
            candidate_answer,
        )

        # Store performance
        interview.add_performance(
            question=current_question["question_en"],
            candidate_answer=candidate_answer,
            evaluation=evaluation["evaluation"],
            strengths=evaluation["strengths"],
            improvements=evaluation["improvements"],
            score=evaluation["score"],
        )

        # Speak evaluation
        response_text = (
            f"{evaluation['evaluation']} "
        )

        # Weak answer -> improvement guidance
        if evaluation["score"] < 5:

            improvement_points = ", ".join(
                evaluation["improvements"]
            )

            response_text += (
                f"You can improve by focusing on: "
                f"{improvement_points}. "
            )

        await worker.queue_frame(
            TTSSpeakFrame(response_text)
        )

        await asyncio.sleep(2)

        # Move next?
        if evaluation["move_next"]:

            interview.advance()

            next_question = interview.get_current_node()

            if next_question:

                new_instruction = generate_grounding_system_instruction()

                context.set_system_instruction(
                    new_instruction
                )

                await asyncio.sleep(1)

                await worker.queue_frame(
                    TTSSpeakFrame(
                        next_question["question_en"]
                    )
                )

            else:
                # FINAL FEEDBACK

                final_feedback = await generate_final_feedback(llm)

                print("\n========== FINAL FEEDBACK ==========")
                print(final_feedback)

                # Speak short summary
                await worker.queue_frame(
                    TTSSpeakFrame(
                        "The interview is complete. "
                        "Generating your final feedback now."
                    )
                )

                await asyncio.sleep(2)

                await worker.queue_frame(
                    TTSSpeakFrame(final_feedback[:500])
                )

                # Save report locally
                with open(
                    "interview_feedback.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(final_feedback)

                print("[INFO] Feedback saved to interview_feedback.txt")

                await asyncio.sleep(3)

                await worker.queue_frame(EndFrame())

        else:
            # Ask follow-up question

            follow_up = (
                "Can you explain that in a bit more detail?"
            )

            await worker.queue_frame(
                TTSSpeakFrame(follow_up)
            )
        print("Starting Pipecat Interview Agent...")
        print(f"Opening Question: {first_question}")

        await runner.run()


if __name__ == "__main__":
    asyncio.run(main())