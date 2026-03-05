import os

import kagglehub
from dotenv import load_dotenv


def download_dataset(competition_name: str):
    kagglehub.competition_download(
        competition_name,
        force_download=True,
        output_dir=os.getenv("DATA_RAW_DIR", "../../data/raw/"),
    )


if __name__ == "__main__":
    load_dotenv()
    download_dataset(os.getenv("KAGGLE_COMPETITION_NAME"))
