import os
import pygsheets
import pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*include_tailing_empty.*",
    category=UserWarning
)   # Already dealing with empty headers.
from dotenv import load_dotenv

def get_castellers() -> pd.DataFrame:
    load_dotenv()
    df = pygsheets.authorize(
        service_file=os.getenv("SERVICE_ACCOUNT_AUTH")
    ).open_by_key(
        os.getenv("FILE_KEY")
    ).worksheet(
        "title", "Base de dades"
    ).get_as_df()
    # Retallar taula
    cut_col = "Funció addicional"
    if cut_col in df.columns:
        cut_idx = df.columns.get_loc(cut_col)
        df = df.iloc[:, :cut_idx]
    else:
        # Fallback: retallar a partir de la primera columna buida
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            first_empty_col = empty_cols[0]
            cut_idx = df.columns.get_loc(first_empty_col)
            df = df.iloc[:, :cut_idx]
    df["assignat"] = False
    return df[df["Assaig"] == "SI"].drop(
        columns=[
            "Inactius", "Assaig", "Sobrenom", "Posicio 3"
        ], errors="ignore"
    )
