import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS


class OpenVoiceTTS:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {self.device.upper()}")

        # === Initialize tone converter ===
        self.converter = ToneColorConverter("checkpoints/converter/config.json", device=self.device)
        self.converter.load_ckpt("checkpoints/converter/checkpoint.pth")

        # === Setup TTS model ===
        self.tts = TTS(language="EN", device=self.device)
        self.speaker_key = "EN-Default"
        self.speaker_id = self.tts.hps.data.spk2id[self.speaker_key]

        speaker_path = f"checkpoints/base_speakers/ses/{self.speaker_key}.pth"
        self.source_se = torch.load(speaker_path, map_location=self.device)

    def synthesize(self, text: str, output_path: str, ref_audio: str) -> str:
        if not text.strip():
            raise ValueError("❌ No text provided for synthesis!")

        if not ref_audio or not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"❌ Reference audio not found: {ref_audio}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_path = output_path.replace(".wav", "_tmp.wav")

        # Generate base audio
        self.tts.tts_to_file(text, self.speaker_id, temp_path)

        # Extract target voice style
        target_se, _ = se_extractor.get_se(ref_audio, self.converter, vad=True)

        # Apply tone/style conversion
        self.converter.convert(
            audio_src_path=temp_path,
            src_se=self.source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell"
        )
        return output_path


# === Test block to run standalone ===
if __name__ == "__main__":
    print("🚀 Starting OpenVoice synthesis test...")

# Make class importable when using `from custom import OpenVoiceTTS`
__all__ = ["OpenVoiceTTS"]
