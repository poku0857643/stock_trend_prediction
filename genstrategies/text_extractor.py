import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Optional
from pathlib import Path
import tiktoken
import warnings
import os
from dotenv import load_dotenv


import pdfplumber

class TextExtractor:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def extract_text_from_pdf(self, pdf_file):
        text = ""
        path = Path(pdf_file)

        if not path.exists() or path.suffix.lower() != ".pdf":
            raise FileFoundError(f"File {path} is not a valid PDF.")

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    def extract_text_from_pdfs(self):
        pdf_texts = {}
        for pdf_file in self.folder_path.glob("*.pdf"):
            try:
                text = self.extract_text_from_pdf(str(pdf_file))
                pdf_texts[pdf_file.name] = text
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")

        return pdf_texts