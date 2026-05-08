from pathlib import Path
import numpy as np

class NeuralRepository:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_representations(self, model_name: str, data: dict):
        file_path = self.output_dir / f"features_{model_name}.npz"
        np.savez_compressed(file_path, **data)