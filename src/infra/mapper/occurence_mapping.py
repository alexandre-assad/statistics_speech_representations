import pandas as pd

class OccurrenceMapper:
    def __init__(self, occurrence_csv_path: str):
        df = pd.read_csv(occurrence_csv_path, sep='\t')
        self.mapping = {}
        
        for _, row in df.iterrows():
            word = row['Word']
            for r in range(1, 7):
                occ_key = f'occ.{r}'
                if pd.notna(row[occ_key]):
                    corp_id = int(row[occ_key])
                    self.mapping[corp_id] = (word, r)

    def get_info(self, corp_id: int):
        return self.mapping.get(corp_id, ("unknown", 0))