import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    """
    Dataset class for sentiment analysis with customizable label classes.
    
    Handles loading text and labels from CSV/Parquet files and tokenizing text.
    """
    def __init__(self, file_path, text_column, label_column, tokenizer, max_seq_length):
        """
        Initialize the sentiment dataset.
        
        Args:
            file_path: Path to the CSV or Parquet file containing the data
            text_column: Name of the column containing the text
            label_column: Name of the column containing the labels
            tokenizer: Tokenizer object to convert text to token IDs
            max_seq_length: Maximum sequence length for padding/truncation
        """
        self.file_path = file_path
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load the dataset from file
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            self.data = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        # Verify columns exist
        if text_column not in self.data.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset.")
        if label_column not in self.data.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset.")
        
        # Check for label type and normalize if needed
        unique_labels = self.data[label_column].unique()
        
        # If labels are strings or non-consecutive integers, map them to integers starting from 0
        if not all(isinstance(label, (int, float)) for label in unique_labels) or \
           not all(0 <= label < len(unique_labels) for label in unique_labels):
            print(f"Converting labels to consecutive integers (0 to {len(unique_labels)-1})...")
            
            # Create a mapping from original labels to integers
            label_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
            
            # Store the mapping for reference
            self.label_mapping = label_mapping
            
            # Apply the mapping
            self.data[label_column] = self.data[label_column].map(label_mapping)
            
            # Print the mapping for users to reference
            print("Label mapping:")
            for original, mapped in label_mapping.items():
                print(f"  {original} -> {mapped}")
        else:
            self.label_mapping = None
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (tokenized_text, label)
        """
        # Get the text and label for this index
        text = str(self.data.iloc[idx][self.text_column])
        label = int(self.data.iloc[idx][self.label_column])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Extract the token IDs and convert label to tensor
        input_ids = encoding.input_ids.squeeze(0)  # Remove batch dimension
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return input_ids, label_tensor

    @classmethod
    def get_label_counts(cls, file_path, label_column):
        """
        Get the count of each unique label in the dataset.
        
        Args:
            file_path: Path to the dataset file
            label_column: Name of the column containing the labels
            
        Returns:
            dict: Mapping from label to count
        """
        # Load the dataset
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        # Count labels
        if label_column in data.columns:
            return data[label_column].value_counts().to_dict()
        else:
            raise ValueError(f"Label column '{label_column}' not found in dataset.")


def get_data_loader(file_path, text_column, label_column, tokenizer, max_seq_length, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the sentiment dataset.
    
    Args:
        file_path: Path to the dataset file
        text_column: Name of the column containing the text
        label_column: Name of the column containing the labels
        tokenizer: Tokenizer object for text processing
        max_seq_length: Maximum sequence length for padding/truncation
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = SentimentDataset(
        file_path=file_path,
        text_column=text_column,
        label_column=label_column,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # Handle cases where the dataset size is not divisible by batch_size
        drop_last=False
    )