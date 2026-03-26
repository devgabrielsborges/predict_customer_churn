import os
from pathlib import Path

import kagglehub
from datasets import load_dataset
from dotenv import load_dotenv


def download_dataset(competition_name: str):
    if os.getenv("DATASET_SOURCE") == "kaggle":
        kagglehub.competition_download(
                competition_name,
                force_download=True,
                output_dir=os.getenv("DATA_RAW_DIR", "../../data/raw/"),
            )
    elif os.getenv("DATASET_SOURCE") == "hf":
        hf_dataset = os.getenv("HF_DATASET")
        dataset = load_dataset(hf_dataset, split="train")
        output_dir = Path(os.getenv("DATA_RAW_DIR", "../../data/raw/"))
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_pandas().to_csv(output_dir / "train.csv")




if __name__ == "__main__":
    load_dotenv()
    download_dataset(os.getenv("KAGGLE_COMPETITION_NAME"))
