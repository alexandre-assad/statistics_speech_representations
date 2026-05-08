import argparse
import torch
import numpy as np
from pathlib import Path

from src.infra.repository.csv_token_repository import CSVTokenRepository
from src.infra.repository.neural_repository import NeuralRepository
from src.infra.extractor.hugging_face_neural_extractor import HuggingFaceNeuralExtractor

def run_neural_extraction(model_name, layers, output_path):
    repo = CSVTokenRepository(Path("data/interim/tokens.csv"))
    neural_repo = NeuralRepository(Path("data/processed"))
    
    model_map = {
        "whisper-medium": "openai/whisper-medium",
        "xlsr-53": "facebook/wav2vec2-large-xlsr-53"
    }
    
    extractor = HuggingFaceNeuralExtractor(model_map[model_name])
    tokens = repo.load_all_tokens()
    
    audio_groups = {}
    for t in tokens:
        if t.file_path not in audio_groups:
            audio_groups[t.file_path] = []
        audio_groups[t.file_path].append(t)

    all_representations = {}
    print(f"Extraction with {model_name} on {len(audio_groups)} files")
    
    for audio_path, group_tokens in audio_groups.items():
        if not Path(audio_path).exists():
            continue
            
        file_results = extractor.extract_batch(group_tokens, audio_path, layers)
        all_representations.update(file_results)

    neural_repo.save_representations(model_name, all_representations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_neural_extraction(args.model, args.layers, args.output)