import os
# os.environ["USE_MECAB"] = "0"
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# === CONFIGURATION ===
CHECKPOINTS_DIR = 'checkpoints'  # Make sure this exists with /converter and /base_speakers
CONVERTER_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, 'converter')
BASE_SPEAKER_DIR = os.path.join(CHECKPOINTS_DIR, 'base_speakers', 'ses')
REFERENCE_VOICE = 'resources/007.mp3'  # Put your .mp3 here
OUTPUT_DIR = 'output'
TEXT = """Hello. I am very happy to be part of this demonstration today.
          Actually, voice cloning technology has come a very long way. It is truly surprising — how just a few seconds of voice recording can reproduce someone’s voice with such accuracy.

          In this test, what you are hearing is not my real voice. This is an artificial voice, created using advanced AI models. Isn’t that very interesting?

          People are using this technology for many purposes — storytelling, dubbing, education, and even accessibility. So the uses are quite wide, you see.

          I hope this sample gives you a clear idea of how natural and expressive these AI voices can sound.

          Thank you so much for listening. Kindly continue exploring this technology. It is very promising!"""
LANGUAGE = "EN"
SPEAKER_KEY = "Huzaifah"  # Must match pth file in /ses/

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# === LOAD TONE COLOR CONVERTER ===
tone_color_converter = ToneColorConverter(os.path.join(CONVERTER_CKPT_DIR, 'config.json'), device=device)
tone_color_converter.load_ckpt(os.path.join(CONVERTER_CKPT_DIR, 'checkpoint.pth'))

# === EXTRACT TARGET SPEAKER EMBEDDING ===
print("🔍 Extracting speaker style from reference voice...")
target_se, _ = se_extractor.get_se(REFERENCE_VOICE, tone_color_converter, vad=True)

# === LOAD BASE SPEAKER AND SYNTHESIZE SPEECH ===
print(f"🎙️ Synthesizing base audio with voice style: {SPEAKER_KEY}")
tts_model = TTS(language=LANGUAGE, device=device)
speaker_ids = tts_model.hps.data.spk2id

# Normalize speaker key to lowercase for consistency
speaker_key_lower = SPEAKER_KEY.lower()

# Dynamically add speaker key if missing
if speaker_key_lower not in speaker_ids:
    max_id = max(speaker_ids.values())
    speaker_ids[speaker_key_lower] = max_id + 1
    print(f"Added new speaker '{speaker_key_lower}' with ID {speaker_ids[speaker_key_lower]}")

speaker_id = speaker_ids[speaker_key_lower]

# Load the source speaker embedding with lowercase filename
source_se_path = os.path.join(BASE_SPEAKER_DIR, f"{speaker_key_lower}.pth")
if not os.path.isfile(source_se_path):
    raise FileNotFoundError(f"Embedding file not found: {source_se_path}")
source_se = torch.load(source_se_path, map_location=device)

# === SYNTHESIZE BASE AUDIO ===
temp_wav = os.path.join(OUTPUT_DIR, "tmp.wav")
tts_model.tts_to_file(TEXT, speaker_id, temp_wav, speed=1.0)

# === CONVERT VOICE STYLE ===
final_output = os.path.join(OUTPUT_DIR, f"cloned_{speaker_key_lower}_pth.wav")
tone_color_converter.convert(
    audio_src_path=temp_wav,
    src_se=source_se,
    tgt_se=target_se,
    output_path=final_output,
    message="@MyShell"
)

print(f"✅ Voice cloning complete! Output saved to: {final_output}")
