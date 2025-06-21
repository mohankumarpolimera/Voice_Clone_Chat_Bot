import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

def extract_and_save_embedding(reference_audio_path, save_embedding_path, converter_checkpoint_dir='checkpoints/converter', device=None):
    # Detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Initialize tone color converter
    tone_color_converter = ToneColorConverter(
        config_path=os.path.join(converter_checkpoint_dir, 'config.json'),
        device=device
    )
    tone_color_converter.load_ckpt(os.path.join(converter_checkpoint_dir, 'checkpoint.pth'))
    print("Loaded tone color converter checkpoint.")

    # Extract embedding from reference audio
    target_se, _ = se_extractor.get_se(reference_audio_path, tone_color_converter, vad=True)
    print(f"Extracted tone color embedding with shape: {target_se.shape}")

    # Make sure save directory exists
    os.makedirs(os.path.dirname(save_embedding_path), exist_ok=True)

    # Save embedding as .pth
    torch.save(target_se, save_embedding_path)
    print(f"Saved tone color embedding to: {save_embedding_path}")


if __name__ == '__main__':
    # === EDIT THESE PATHS BEFORE RUNNING ===
    reference_audio = 'resources/007.mp3'  # Your reference audio path here
    save_path = 'checkpoints/base_speakers/ses/Huzaifah.pth'  # Where to save the embedding

    extract_and_save_embedding(reference_audio, save_path)
