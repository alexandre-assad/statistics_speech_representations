import argparse
from pathlib import Path
import re

from src.infra.parser.textgrid_parser import TextGridParser
from src.infra.repository.csv_token_repository import CSVTokenRepository
from src.infra.mapper.occurence_mapping import OccurrenceMapper


def run_parsing(
    corpus_dir: str, metadata_path: str, output_path: str, occurrence_path: str
):
    parser = TextGridParser(metadata_path=Path(metadata_path))
    mapper = OccurrenceMapper(occurrence_path)
    repo = CSVTokenRepository(output_path=Path(output_path))

    all_tokens = []
    valid_speakers = parser.metadata["spk"].unique()
    print(valid_speakers)

    for tg_file in Path(corpus_dir).rglob("*.TextGrid"):
        speaker_id = next(
            (s for s in valid_speakers if s.lower() in tg_file.name), None
        )
        if not speaker_id:
            print("No speaker id")
            continue

        match = re.search(r"(?:FR|RU)corp(\d+)", tg_file.name)
        if not match:
            print("no match")
            continue

        corp_id = int(match.group(1))
        word, repetition = mapper.get_info(corp_id)

        tokens = parser.parse_file(word, tg_file, speaker_id, word, repetition)

        for t in tokens:
            t.word = word

        all_tokens.extend(tokens)

    print(all_tokens[0])
    repo.save_tokens(all_tokens)
    print(f"End parsing : {len(all_tokens)} extracted tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--occurrence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_parsing(args.corpus_dir, args.metadata, args.output, args.occurrence)
