import logging

import pandas as pd
from torch.utils.data import Dataset


class WebDocument(Dataset):
    def __init__(self, document_list: str, size: int = 0):
        self.df = pd.read_csv(document_list)

        if size > 0:
            self.df = self.df[:size]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


def main():
    logging.basicConfig(level=logging.DEBUG)

    ds = WebDocument("./document_list.csv")

    logging.info(ds[0])


if __name__ == "__main__":
    main()
