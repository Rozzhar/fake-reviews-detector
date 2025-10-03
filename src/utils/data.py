import re
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from typing import Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Class to handle data loading, preprocessing, and conversion to Hugging Face Dataset.
    """
    def __init__(self, file_path: str):
        """
        Initialize the DataProcessor with the file path.

        Args:
            file_path (str): Path to the raw CSV file.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Dataset loaded successfully from {self.file_path}. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning: remove punctuation and numbers, convert to lowercase.

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text.
        """
        if pd.isna(text):
            return ""
        return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame: map labels, clean text.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        # Map labels: 'OR' -> 0 (real), 'CG' -> 1 (fake)
        df['label'] = df['label'].map({'OR': 0, 'CG': 1})
        if df['label'].isnull().any():
            logger.warning("Some labels could not be mapped. Check dataset integrity.")

        # Clean text
        df['text_'] = df['text_'].apply(self.clean_text)
        logger.info("Data preprocessing completed.")
        return df

    def split_and_convert(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Split data into train/test and convert to Hugging Face Datasets.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
            test_size (float): Test split ratio.

        Returns:
            Tuple[Dataset, Dataset]: Train and eval datasets.
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=42
        )
        logger.info(f"Data split: Train {train_df.shape[0]}, Test {test_df.shape[0]}")

        train_dataset = Dataset.from_pandas(train_df[['text_', 'label']])
        eval_dataset = Dataset.from_pandas(test_df[['text_', 'label']])
        return train_dataset, eval_dataset
