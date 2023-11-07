import logging

import pandas as pd
from torch.utils.data import Dataset


class WebDocument(Dataset):
    def __init__(self, document_list: str):
        self.df = pd.read_csv(document_list)

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
