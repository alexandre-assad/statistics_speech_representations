import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from src.domain.abstract.neural_extractor_service import NeuralExtractorService

class HuggingFaceNeuralExtractor(NeuralExtractorService):
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract_batch(self, tokens, audio_path, layers):
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        all_layers = outputs.hidden_states 
        results = {}

        for token in tokens:
            start_idx = int(token.onset * 50) 
            end_idx = int(token.offset * 50)
            
            token_layers = {}
            for l in layers:
                layer_data = all_layers[l][0, start_idx:end_idx, :]
                token_layers[f"layer_{l}"] = layer_data.mean(dim=0).cpu().numpy()
            
            results[token.token_id] = token_layers
            
        return results