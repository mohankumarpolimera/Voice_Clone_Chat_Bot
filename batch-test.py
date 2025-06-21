import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# === CONFIGURATION ===
CHECKPOINTS_DIR = 'checkpoints'
CONVERTER_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, 'converter')
BASE_SPEAKER_DIR = os.path.join(CHECKPOINTS_DIR, 'base_speakers', 'ses')
REFERENCE_VOICE = 'resources/007.mp3'  # Your voice sample
OUTPUT_DIR = 'output'
TEXT = "Hello. I am testing this voice clone to check which speaker sounds most like me."

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load the tone color converter ===
converter = ToneColorConverter(os.path.join(CONVERTER_CKPT_DIR, 'config.json'), device=device)
converter.load_ckpt(os.path.join(CONVERTER_CKPT_DIR, 'checkpoint.pth'))

# === Extract target speaker embedding ===
print("🔍 Extracting your voice style...")
target_se, _ = se_extractor.get_se(REFERENCE_VOICE, converter, vad=True)

# === Load TTS model ===
tts = TTS(language="EN", device=device)
speaker_ids = tts.hps.data.spk2id

# === Run through each speaker file ===
for filename in os.listdir(BASE_SPEAKER_DIR):
    if not filename.endswith(".pth"):
        continue

    speaker_key = filename.replace(".pth", "")
    output_path = os.path.join(OUTPUT_DIR, f"Huzaifah_cloned_{speaker_key}.wav")
    speaker_path = os.path.join(BASE_SPEAKER_DIR, filename)

    print(f"🎙️ Synthesizing with base speaker: {speaker_key}")
    source_se = torch.load(speaker_path, map_location=device)
    speaker_id = speaker_ids[speaker_key] if speaker_key in speaker_ids else 0

    tmp_path = os.path.join(OUTPUT_DIR, "tmp.wav")
    tts.tts_to_file(TEXT, speaker_id, tmp_path, speed=1.0)

    converter.convert(
        audio_src_path=tmp_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell"
    )

print("✅ Done! Check the 'output/' folder to listen and compare each result.")
