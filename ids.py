from melo.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Choose a language to inspect (change to "FR", "ES", etc. to check other models)
language = "EN"

# Load the model
tts = TTS(language=language, device=device)

# Print available speaker IDs
print("\n🗣 Registered Speakers for language:", language)
for name, idx in tts.hps.data.spk2id.items():
    print(f" - {name}: ID {idx}")
