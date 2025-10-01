# Fake Product Review Detector

## Description
This proyect detects fake product reviews using Transformers (DistilBERT). The model is customed trained using the Kaggle Dataset "Fake Reviews"

## Installation
1. Clone the repo: `git clone <url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Kaggle dataset and unzzip it in data/raw/

## Uso
- Training: `python src/train.py`
- API: `uvicorn src.api.app:app --reload`
- Go to http://localhost:8000/docs and try the API.

## Dataset
https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset