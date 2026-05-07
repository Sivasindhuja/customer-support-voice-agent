import requests
import re
import os
import threading
import time
import json
import pygame
import tempfile
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
from dotenv import load_dotenv

# Google GenAI imports
from google import genai
from google.genai import types

load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize clients
dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-helios-en'
HEADERS = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

# State Management
conversation_memory = []
mute_microphone = threading.Event()
REPLAY_FILE = "replay.log"

def get_active_prompt(role="receptionist", version="v1.1.0"):
    try:
        with open('prompts.json', 'r') as f:
            data = json.load(f)
        return data[role][version]
    except Exception as e:
        print(f"Prompt Load Error: {e}")
        return "You are a helpful assistant."

def segment_text_by_sentence(text):
    sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', text)
    boundaries_indices = [boundary.start() for boundary in sentence_boundaries]
    segments = []
    start = 0
    for boundary_index in boundaries_indices:
        segments.append(text[start:boundary_index + 1].strip())
        start = boundary_index + 1
    segments.append(text[start:].strip())
    return [s for s in segments if s]

def synthesize_audio(text):
    payload = {"text": text}
    try:
        # Timeout added for resilience
        r = requests.post(DEEPGRAM_TTS_URL, headers=HEADERS, json=payload, timeout=5.0)
        return r.content
    except requests.exceptions.Timeout:
        print("TTS Timeout - Service unreachable.")
        return None

def play_audio_data(audio_data):
    if not audio_data: return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def visualize_budget(m):
    # Total perceived latency from end of speech to start of audio
    print("\n" + "="*30)
    print(f"LATENCY BREAKDOWN (Total: {m['total_ms']:.0f}ms)")
    print(f"  1. ASR (VAD -> Text):     {m['asr_ms']:>6.0f}ms")
    print(f"  2. LLM (Text -> Token 1): {m['llm_ttft_ms']:>6.0f}ms")
    print(f"  3. TTS (Text -> Byte 1):  {m['tts_ttfb_ms']:>6.0f}ms")
    print(f"  4. Overhead/Misc:         {m['overhead_ms']:>6.0f}ms")
    print("="*30)


# --- UPGRADED: Streaming AI Logic ---
def process_turn(utterance, start_time, asr_end, microphone):
    try:
        llm_start = time.perf_counter()
        
        conversation_memory.append(
            types.Content(role="user", parts=[types.Part.from_text(text=utterance)])
        )
        
        active_system_prompt = get_active_prompt("receptionist", "v1.1.0")
        
        # --- PHASE 1: STREAMING ENABLED ---
        response_stream = gemini_client.models.generate_content_stream(
            model="gemini-3-flash-preview", 
            contents=conversation_memory,
            config=types.GenerateContentConfig(
                system_instruction=active_system_prompt
            )
        )
        
        full_text = ""
        sentence_buffer = ""
        first_token_received = False
        
        llm_ttft_ms = 0
        tts_ttfb_ms = 0
        perceived_latency_end = 0
        
        for chunk in response_stream:
            # Capture true Time To First Token
            if not first_token_received:
                llm_ttft_ms = (time.perf_counter() - llm_start) * 1000
                first_token_received = True
            
            chunk_text = chunk.text
            full_text += chunk_text
            sentence_buffer += chunk_text

            # Check if we have a complete sentence to speak right now
            if any(punc in chunk_text for punc in ".!?"):
                sentences = segment_text_by_sentence(sentence_buffer)
                
                if len(sentences) > 1:
                    to_process = sentences[:-1] # Full sentences ready for TTS
                    sentence_buffer = sentences[-1] # Keep the trailing fragment
                else:
                    to_process = sentences
                    sentence_buffer = ""

                for seg in to_process:
                    tts_start = time.perf_counter()
                    audio = synthesize_audio(seg)
                    
                    # Capture true Total Latency when the FIRST audio byte is ready
                    if tts_ttfb_ms == 0 and audio:
                        perceived_latency_end = time.perf_counter() 
                        tts_ttfb_ms = (perceived_latency_end - tts_start) * 1000
                    
                    play_audio_data(audio)

        # Process any leftover text after the stream finishes
        if sentence_buffer.strip():
            tts_start = time.perf_counter()
            audio = synthesize_audio(sentence_buffer)
            if tts_ttfb_ms == 0 and audio:
                perceived_latency_end = time.perf_counter()
                tts_ttfb_ms = (perceived_latency_end - tts_start) * 1000
            play_audio_data(audio)

        conversation_memory.append(
            types.Content(role="model", parts=[types.Part.from_text(text=full_text)])
        )

        # Metrics Calculation
        asr_val = (asr_end - start_time) * 1000
        total_val = (perceived_latency_end - start_time) * 1000
        overhead_val = total_val - (asr_val + llm_ttft_ms + tts_ttfb_ms)

        metrics = {
            "asr_ms": asr_val,
            "llm_ttft_ms": llm_ttft_ms,
            "tts_ttfb_ms": tts_ttfb_ms,
            "overhead_ms": max(0, overhead_val), 
            "total_ms": total_val
        }
        
        visualize_budget(metrics)
        return metrics

    except Exception as e:
        print(f"Streaming Error: {e}")
        if len(conversation_memory) > 0 and conversation_memory[-1].role == "user":
            conversation_memory.pop()
        play_audio_data(synthesize_audio("I'm sorry, I'm having a bit of trouble connecting to my brain. Could you repeat that?"))
        return None
    
    finally:
        time.sleep(0.3)
        microphone.unmute()
        mute_microphone.clear()

def main():
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.websocket.v("1")
        is_finals = []

        def on_message(self, result, **kwargs):
            nonlocal is_finals
            if mute_microphone.is_set(): return
            
            transcript = result.channel.alternatives[0].transcript
            if len(transcript) == 0: return

            if result.is_final:
                is_finals.append(transcript)
                
                if result.speech_final:
                    # --- Start Latency Tracking ---
                    start_time = time.perf_counter()
                    utterance = " ".join(is_finals).strip()
                    is_finals = []
                    
                    # Log for Replay Mode
                    with open(REPLAY_FILE, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | USER: {utterance}\n")

                    print(f"User: {utterance}")
                    
                    # 1. ASR Finish
                    asr_end = time.perf_counter()

                    # Immediately lock the microphone on the main thread
                    mute_microphone.set()
                    microphone.mute()
                    
                    # Pass the workload to the background thread!
                    threading.Thread(
                        target=process_turn, 
                        args=(utterance, start_time, asr_end, microphone)
                    ).start()

        # Deepgram Handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, lambda self, error, **kw: print(f"Error: {error}"))

        options = LiveOptions(
            model="nova-2", language="en-US", smart_format=True,
            encoding="linear16", channels=1, sample_rate=16000,
            interim_results=True, utterance_end_ms="1000", vad_events=True, endpointing=500,
        )

        if not dg_connection.start(options):
            print("Failed to connect to Deepgram")
            return

        microphone = Microphone(dg_connection.send)
        microphone.start()
        print("Agent Active. Press Enter to stop.")
        input("")
        microphone.finish()
        dg_connection.finish()

    except Exception as e:
        print(f"System Error: {e}")

# --- TESTING SUITE ---
class MockMicrophone:
    def mute(self): pass
    def unmute(self): pass

TEST_QUERIES = [
    {"id": "short_greeting", "text": "Hello."},
    {"id": "medium_booking", "text": "I would like to book a table for 4 people tonight at 8 PM."},
    {"id": "long_menu", "text": "Can you tell me all the appetizers you have and their prices?"}
]

def run_local_baseline():
    results = []
    print("Starting Baseline Test Suite...")
    
    for query in TEST_QUERIES:
        print(f"\nRunning: {query['id']}")
        conversation_memory.clear() 
        
        start_time = time.perf_counter()
        asr_end = start_time + 0.05 # Simulate 50ms ASR completion
        
        metrics = process_turn(query['text'], start_time, asr_end, MockMicrophone())
        
        if metrics:
            results.append({
                "query": query["id"],
                "total_latency_ms": metrics["total_ms"],
                "llm_ttft_ms": metrics["llm_ttft_ms"],
                "tts_ttfb_ms": metrics["tts_ttfb_ms"]
            })
            
            # Wait 60 seconds to avoid Gemini Free Tier limits (429/503 errors)
            print("Waiting 60 seconds for quota cooldown...")
            time.sleep(60)
        else:
            print("Test failed, waiting 30 seconds before retrying...")
            time.sleep(30)

    with open("baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results written to baseline_metrics.json")

if __name__ == "__main__":
    # Run the baseline tester instead of the live mic
    run_local_baseline()