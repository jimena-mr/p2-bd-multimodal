import os
from pathlib import Path
import pandas as pd
import streamlit as st

def list_tables(base_path="data"):
    tables = []
    for path in Path(base_path).iterdir():
        if path.is_dir() and (path / "metadata.csv").exists():
            tables.append(path.name)
    return sorted(tables)
