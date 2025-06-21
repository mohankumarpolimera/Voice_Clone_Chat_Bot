import os
os.environ["USE_MECAB"] = "0"
import torch
import sounddevice as sd
import soundfile as sf
import whisper
import openai
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# ==== CONFIGURATION ====
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
REFERENCE_VOICE = "resources/007.mp3"
SPEAKER_KEY = "EN-Default"
LANGUAGE = "EN"
CHECKPOINTS_DIR = "checkpoints"
BASE_SPEAKER_DIR = os.path.join(CHECKPOINTS_DIR, "base_speakers", "ses")
CONVERTER_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "converter")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REC_AUDIO = os.path.join(OUTPUT_DIR, "user_input.wav")
TEMP_TTS = os.path.join(OUTPUT_DIR, "tmp.wav")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "cloned_reply.wav")

# ==== INIT MODELS ====
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base")
tone_color_converter = ToneColorConverter(os.path.join(CONVERTER_CKPT_DIR, "config.json"), device=device)
tone_color_converter.load_ckpt(os.path.join(CONVERTER_CKPT_DIR, "checkpoint.pth"))

print("🔍 Extracting speaker style from reference...")
target_se, _ = se_extractor.get_se(REFERENCE_VOICE, tone_color_converter, vad=True)

tts_model = TTS(language=LANGUAGE, device=device)
speaker_ids = tts_model.hps.data.spk2id
if SPEAKER_KEY not in speaker_ids:
    raise ValueError(f"Speaker '{SPEAKER_KEY}' not found.")
speaker_id = speaker_ids[SPEAKER_KEY]
source_se_path = os.path.join(BASE_SPEAKER_DIR, f"{SPEAKER_KEY}.pth")
source_se = torch.load(source_se_path, map_location=device)

# ==== FUNCTIONS ====

def record_audio(filename=REC_AUDIO, duration=5, samplerate=44100):
    print("🎤 Speak now...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, recording, samplerate)
    print("✅ Recorded.")

def transcribe_audio(filename=REC_AUDIO):
    print("🧠 Transcribing...")
    result = whisper_model.transcribe(filename)
    print(f"🗣️ You said: {result['text']}")
    return result["text"]

def query_gpt(prompt):
    print("🤖 Querying GPT-4...")
    completion = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": "You are a helpful voice assistant."},
                  {"role": "user", "content": prompt}]
    )
    reply = completion.choices[0].message.content
    print(f"🤖 GPT says: {reply}")
    return reply

def speak_reply(text):
    print("🗣️ Synthesizing voice...")
    tts_model.tts_to_file(text, speaker_id, TEMP_TTS, speed=1.0)
    tone_color_converter.convert(
        audio_src_path=TEMP_TTS,
        src_se=source_se,
        tgt_se=target_se,
        output_path=FINAL_OUTPUT,
        message="@OpenVoice"
    )
    os.system(f"ffplay -nodisp -autoexit {FINAL_OUTPUT}")

# ==== MAIN LOOP ====
def main():
    print("🎧 One-on-One Voice Chat (say 'exit' to quit)")
    while True:
        record_audio()
        user_text = transcribe_audio()
        if user_text.strip().lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting.")
            break
        reply = query_gpt(user_text)
        speak_reply(reply)

if __name__ == "__main__":
    main()
