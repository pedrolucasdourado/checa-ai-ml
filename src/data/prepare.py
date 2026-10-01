"""Harmoniza as 3 bases brutas para o esquema canonico (text, label, source).

label: 0 = Fake, 1 = Real. Espelha a harmonizacao de notebooks/eda_comparativa.ipynb.
"""
from pathlib import Path

import pandas as pd
import yaml

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/dataset.csv")


def load_fakerecogna() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "FakeRecogna.xlsx").dropna(subset=["Classe"])
    return pd.DataFrame({
        "text": df["Noticia"].fillna(df["Titulo"]).astype(str),
        "label": df["Classe"].astype(int),
        "source": "FakeRecogna",
    })


def load_faketruebr() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "FakeTrueBr_corpus.csv")
    fake = pd.DataFrame({"text": df["fake"].astype(str), "label": 0, "source": "FakeTrueBr"})
    true = pd.DataFrame({"text": df["true"].astype(str), "label": 1, "source": "FakeTrueBr"})
    return pd.concat([fake, true], ignore_index=True)


def load_fakewhatsapp() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "fakeWhatsApp.BR_2018.csv", low_memory=False)
    df = df[df["misinformation"].isin([0, 1])]
    return pd.DataFrame({
        "text": df["text"].astype(str),
        "label": df["misinformation"].map({1: 0, 0: 1}),  # na base original 1 = desinformacao
        "source": "fakeWhatsApp",
    })


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.concat([load_fakerecogna(), load_faketruebr(), load_fakewhatsapp()], ignore_index=True)
    df = df[df["text"].str.split().str.len() > params["min_words"]]
    df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=params["seed"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df):,} amostras -> {OUT_PATH}")


if __name__ == "__main__":
    main()
