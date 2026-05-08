import argparse
from pathlib import Path

from numpy import full

from src.infra.repository.csv_token_repository import CSVTokenRepository
from src.infra.extractor.praat_acoustic_extractor import PraatAcousticExtractor

def run_extraction(input_csv: str, audio_dir: str, output_csv: str):
    repo = CSVTokenRepository(output_path=Path(input_csv))
    extractor = PraatAcousticExtractor()
    
    tokens = repo.load_all_tokens()
    
    for token in tokens:
        full_audio_path = Path(token.file_path)
        if full_audio_path.exists():
            try:
                token.acoustic = extractor.extract(token, str(full_audio_path))
            except Exception as e:
                print(f"Error on token {token.token_id}: {e}")
        else:
            print(full_audio_path)
    
    repo.output_path = Path(output_csv)
    repo.save_tokens(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_extraction(args.input, args.audio_dir, args.output)