import os
import logging
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Tuple
from datasets import Dataset
from utils.data import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class FakeReviewTrainer:
    """
    Class to handle the training pipeline for the fake review detector using Transformers.
    """
    def __init__(self, config_path: str):
        """
        Initialize the trainer with configuration.

        Args:
            config_path (str): Path to config.yaml.
        """
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.trainer = None

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to config file.

        Returns:
            Dict: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def load_model_and_tokenizer(self):
        """
        Load tokenizer and model from pretrained.
        """
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        logger.info(f"Model and tokenizer loaded: {model_name}")

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        """
        Prepare data using DataProcessor.

        Returns:
            Tuple[Dataset, Dataset]: Tokenized train and eval datasets.
        """
        raw_path = self.config['data']['raw_path']
        processor = DataProcessor(raw_path)
        df = processor.load_data()
        df = processor.preprocess_data(df)
        train_dataset, eval_dataset = processor.split_and_convert(df)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(examples['text_'], padding='max_length', truncation=True, max_length=128)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        logger.info("Datasets tokenized.")
        return tokenized_train, tokenized_eval

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset):
        """
        Setup Hugging Face Trainer.

        Args:
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset): Evaluation dataset.
        """
        output_dir = self.config['model']['output_dir']
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.from_numpy(logits), dim=-1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            return {'accuracy': acc, 'f1': f1}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        logger.info("Trainer setup completed.")

    def train(self):
        """
        Run the training process.
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer first.")
        self.trainer.train()
        logger.info("Training completed.")

    def save_model(self):
        """
        Save the best model.
        """
        output_dir = self.config['model']['output_dir'] + 'best_model'
        self.trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")


def main():
    # Resolve config path relative to this file
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    trainer = FakeReviewTrainer(config_path)
    trainer.load_model_and_tokenizer()
    train_dataset, eval_dataset = trainer.prepare_data()
    trainer.setup_trainer(train_dataset, eval_dataset)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
