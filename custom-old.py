import os
# os.environ["USE_MECAB"] = "0"
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
SPEAKER_KEY = "EN-Default"  # Must match pth file in /ses/

# === SETUP ===s
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

if SPEAKER_KEY not in speaker_ids:
    raise ValueError(f"Speaker '{SPEAKER_KEY}' not found in speaker IDs. Check your .pth filenames.")

speaker_id = speaker_ids[SPEAKER_KEY]
source_se_path = os.path.join(BASE_SPEAKER_DIR, f"{SPEAKER_KEY}.pth")
source_se = torch.load(source_se_path, map_location=device)

# === SYNTHESIZE BASE AUDIO ===
temp_wav = os.path.join(OUTPUT_DIR, "tmp.wav")
tts_model.tts_to_file(TEXT, speaker_id, temp_wav, speed=1.0)

# === CONVERT VOICE STYLE ===
final_output = os.path.join(OUTPUT_DIR, f"cloned_EN_Default.wav")
tone_color_converter.convert(
    audio_src_path=temp_wav,
    src_se=source_se,
    tgt_se=target_se,
    output_path=final_output,
    message="@MyShell"
)

print(f"✅ Voice cloning complete! Output saved to: {final_output}")
