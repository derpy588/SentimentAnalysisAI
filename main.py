import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from model import SentimentClassifier # Assuming model.py contains SentimentClassifier
from SentimentDataset import get_data_loader # Assuming SentimentDataset.py contains get_data_loader
import numpy as np
import json
import pandas as pd
import time # Import the time module

# Import DirectML backend if available
_directml_available = False
try:
    import torch.backends.directml
    _directml_available = torch.backends.directml.is_available()
except ImportError:
    _directml_available = False
    print("Warning: torch_directml not installed. DirectML backend will not be available.")

class SentimentAnalysisSystem:
    """
    Main class for the sentiment analysis system that handles:
    - Model initialization
    - Training
    - Evaluation
    - Inference
    - Visualization
    - Configuration management
    - Device management (CPU, CUDA, DirectML)
    """
    def __init__(self, model_config=None):
        """
        Initialize the sentiment analysis system.

        Args:
            model_config: Configuration dictionary for the model (optional)
        """
        # Set default configuration values
        self.config = {
            'train_file': "data/compile_data/final/train.csv",
            'val_file': "data/compile_data/final/validation.csv",
            'test_file': "data/compile_data/final/test.csv",
            'text_column': "text",
            'label_column': "sentiment",
            'max_seq_length': 32,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_classes': 3,
            'class_names': ["Negative", "Neutral", "Positive"],
            'embedding_dim': 128,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout_rate': 0.3,
            'use_attention': True,
            'model_save_path': "models/sentiment_model.pt",
            'config_save_path': "models/sentiment_config.json",
            'log_interval': 200 # New configuration for logging frequency
        }

        # Update with custom config if provided
        if model_config:
            self.config.update(model_config)

        # Ensure required directories exist
        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)

        # Initialize the tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")

        # Initialize device (will be set later by detect_and_set_device)
        self.device = torch.device("cpu")
        print(f"Initial device set to: {self.device}")

        # Initialize the model (will be moved to the selected device later)
        self._initialize_model()

        # Prepare data loaders if files exist
        self.train_loader = self._create_data_loader(self.config['train_file']) if os.path.exists(self.config['train_file']) else None
        self.val_loader = self._create_data_loader(self.config['val_file']) if os.path.exists(self.config['val_file']) else None
        self.test_loader = self._create_data_loader(self.config['test_file']) if os.path.exists(self.config['test_file']) else None

        # Initialize tracking variables
        self.epoch_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def _initialize_model(self):
        """Initialize or reinitialize the model with current configuration and move to device."""
        # Initialize the model
        self.model = SentimentClassifier(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=self.config['num_classes'],
            num_layers=self.config['num_layers'],
            dropout_rate=self.config['dropout_rate'],
            use_attention=self.config['use_attention']
        )

        # Move model to the currently set device
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")

        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")

        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

    def detect_available_devices(self):
        """
        Detect available computation devices (CPU, CUDA, DirectML).

        Returns:
            list: A list of available device names (e.g., ['cpu', 'cuda', 'dml']).
        """
        # Declare _directml_available as global to access the module-level variable
        global _directml_available

        available_devices = ["cpu"] # CPU is always available

        if torch.cuda.is_available():
            available_devices.append("cuda")
            print(f"CUDA available: {torch.cuda.device_count()} device(s)")

        if _directml_available:
            # DirectML devices are typically indexed starting from 0
            try:
                # Check if any DirectML device is actually initialized
                dml_device_count = torch.backends.directml.device_count()
                if dml_device_count > 0:
                     available_devices.append("dml")
                     print(f"DirectML available: {dml_device_count} device(s)")
            except Exception as e:
                 print(f"Error checking DirectML devices: {e}")
                 _directml_available = False # Disable DML if there's an error

        # Note: ZLUDA support is less direct in PyTorch. It often works by
        # intercepting CUDA calls. If ZLUDA is set up correctly, it *might*
        # appear as a CUDA device. Explicit ZLUDA detection beyond checking
        # CUDA availability is not standard in PyTorch's device API.

        return available_devices

    def set_device(self, device_name):
        """
        Set the computation device for the system.

        Args:
            device_name (str): The name of the device ('cpu', 'cuda', 'dml').
        """
        # Declare _directml_available as global to access the module-level variable
        global _directml_available

        if device_name == "cuda" and not torch.cuda.is_available():
            print("Error: CUDA is not available on this system. Setting device to CPU.")
            self.device = torch.device("cpu")
        elif device_name == "dml" and not _directml_available:
             print("Error: DirectML is not available on this system. Setting device to CPU.")
             self.device = torch.device("cpu")
        elif device_name not in ["cpu", "cuda", "dml"]:
             print(f"Warning: Unknown device '{device_name}'. Setting device to CPU.")
             self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)

        # Move the model to the newly set device
        self.model.to(self.device)
        print(f"Computation device set to: {self.device}")


    def save_config(self):
        """Save the current configuration to a JSON file."""
        # Exclude device information as it's system-specific
        config_to_save = self.config.copy()
        # No need to explicitly remove device as it's not in self.config initially

        with open(self.config['config_save_path'], 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print(f"Configuration saved to {self.config['config_save_path']}")

    def load_config(self, config_path=None):
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file (optional)
        """
        if config_path is None:
            config_path = self.config['config_save_path']

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            print(f"Configuration loaded from {config_path}")

            # Reinitialize model with the loaded configuration
            # Model will be moved to the *currently set* device in _initialize_model
            self._initialize_model()

            # Reload data loaders
            self.train_loader = self._create_data_loader(self.config['train_file']) if os.path.exists(self.config['train_file']) else None
            self.val_loader = self._create_data_loader(self.config['val_file']) if os.path.exists(self.config['val_file']) else None
            self.test_loader = self._create_data_loader(self.config['test_file']) if os.path.exists(self.config['test_file']) else None
        else:
            print(f"Configuration file {config_path} not found.")

    def _create_data_loader(self, file_path):
        """Create a data loader for the given file path."""
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            return None

        return get_data_loader(
            file_path=file_path,
            text_column=self.config['text_column'],
            label_column=self.config['label_column'],
            tokenizer=self.tokenizer,
            max_seq_length=self.config['max_seq_length'],
            batch_size=self.config['batch_size']
        )

    def set_dataset(self, train_file=None, val_file=None, test_file=None,
                    text_column=None, label_column=None):
        """
        Set new dataset files and column names.

        Args:
            train_file: Path to the training dataset (optional)
            val_file: Path to the validation dataset (optional)
            test_file: Path to the test dataset (optional)
            text_column: Name of the text column (optional)
            label_column: Name of the label column (optional)
        """
        # Update configuration with provided values
        if train_file:
            self.config['train_file'] = train_file
        if val_file:
            self.config['val_file'] = val_file
        if test_file:
            self.config['test_file'] = test_file
        if text_column:
            self.config['text_column'] = text_column
        if label_column:
            self.config['label_column'] = label_column

        # Reinitialize data loaders
        if train_file or text_column or label_column:
            self.train_loader = self._create_data_loader(self.config['train_file'])
        if val_file or text_column or label_column:
            self.val_loader = self._create_data_loader(self.config['val_file'])
        if test_file or text_column or label_column:
            self.test_loader = self._create_data_loader(self.config['test_file'])

        # Check if we need to update class information
        if train_file or label_column:
            self._update_class_information()

    def _update_class_information(self):
        """Update class information from the training dataset."""
        if os.path.exists(self.config['train_file']):
            # Load the dataset
            if self.config['train_file'].endswith('.csv'):
                df = pd.read_csv(self.config['train_file'])
            elif self.config['train_file'].endswith('.parquet'):
                df = pd.read_parquet(self.config['train_file'])
            else:
                print(f"Unsupported file format: {self.config['train_file']}")
                return

            # Get unique labels and count
            if self.config['label_column'] in df.columns:
                unique_labels = sorted(df[self.config['label_column']].unique())
                num_classes = len(unique_labels)

                # Update config
                self.config['num_classes'] = num_classes

                # If we have class names, use them. Otherwise, generate generic names
                if len(self.config['class_names']) != num_classes:
                    self.config['class_names'] = [f"Class {i}" for i in range(num_classes)]

                print(f"Updated class information: {num_classes} classes found")

                # Reinitialize the model with updated number of classes
                # Model will be moved to the currently set device in _initialize_model
                self._initialize_model()
            else:
                print(f"Label column '{self.config['label_column']}' not found in dataset.")

    def train(self, epochs):
        """
        Train the model for the given number of epochs.

        Args:
            epochs: Number of training epochs

        Returns:
            dict: Training history
        """
        # Check if training data is available
        if self.train_loader is None:
            print("Error: Training data not available.")
            return None

        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs on device: {self.device}")
        print(f"{'='*60}")

        best_val_loss = float('inf')

        # Get total number of batches
        total_batches = len(self.train_loader)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            batch_count = 0

            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)

            # Record start time for the epoch
            epoch_start_time = time.time()

            for i, (inputs, labels) in enumerate(self.train_loader):
                # Move data to the selected device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                batch_count += 1

                # --- Progress Tracking ---
                current_time = time.time()
                elapsed_time = current_time - epoch_start_time
                batches_processed = i + 1
                batches_remaining = total_batches - batches_processed

                # Calculate estimated time remaining
                if batches_processed > 0:
                    avg_time_per_batch = elapsed_time / batches_processed
                    estimated_time_remaining = avg_time_per_batch * batches_remaining

                    # Format estimated time remaining
                    etr_minutes = int(estimated_time_remaining // 60)
                    etr_seconds = int(estimated_time_remaining % 60)
                    time_remaining_str = f"{etr_minutes}m {etr_seconds}s"
                else:
                    time_remaining_str = "Calculating..."

                # Print progress for each batch
                print(f"  Batch {batches_processed}/{total_batches} | Remaining: {batches_remaining} | Est. Time Left: {time_remaining_str}", end='\r')

                # Print loss every log_interval batches
                if (i + 1) % self.config['log_interval'] == 0:
                     # Print on a new line to not overwrite the progress bar
                    print(f"\n  Batch {i+1}: Average Loss over last {self.config['log_interval']} batches = {running_loss / self.config['log_interval']:.4f}")
                    running_loss = 0.0

            # Print a newline after the progress bar finishes
            print()

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / batch_count
            self.epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} - Training Loss: {avg_epoch_loss:.4f}")

            # Validation phase
            if self.val_loader is not None:
                val_loss, val_accuracy = self.evaluate(self.val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

                print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

                # Save model if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save model state dictionary (device-agnostic)
                    torch.save(self.model.state_dict(), self.config['model_save_path'])
                    print(f"Model saved to {self.config['model_save_path']}")

                    # Save configuration
                    self.save_config()
            else:
                # If no validation set, save model after each epoch
                # Save model state dictionary (device-agnostic)
                torch.save(self.model.state_dict(), self.config['model_save_path'])
                print(f"Model saved to {self.config['model_save_path']}")

                # Save configuration
                self.save_config()

        # Plot training history
        self.plot_training_history()

        return {
            'train_losses': self.epoch_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

    def evaluate(self, data_loader):
        """
        Evaluate the model on the given data loader.

        Args:
            data_loader: DataLoader with evaluation data

        Returns:
            tuple: (average_loss, accuracy)
        """
        if data_loader is None:
            print("Error: Evaluation data not available.")
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                # Move data to the selected device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def test(self):
        """
        Test the model on the test dataset.

        Returns:
            tuple: (test_loss, test_accuracy, classification_report)
        """
        if self.test_loader is None:
            print("Error: Test data not available.")
            return None

        print("\nEvaluating model on test data...")

        # Load the best model state dictionary (device-agnostic loading)
        if os.path.exists(self.config['model_save_path']):
            # Load the state_dict and then load it into the model
            state_dict = torch.load(self.config['model_save_path'], map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model state loaded from {self.config['model_save_path']} and mapped to {self.device}")
        else:
            print(f"Warning: Model file {self.config['model_save_path']} not found. Using current model on device: {self.device}")

        # Evaluate on test set
        test_loss, test_accuracy = self.evaluate(self.test_loader)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Calculate per-class metrics
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                 # Move data to the selected device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                # Move predictions and labels back to CPU for numpy conversion
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate confusion matrix
        cm = self._confusion_matrix(all_labels, all_preds, self.config['num_classes'])

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("            Predicted")
        print("            " + "  ".join(f"{name[:3]}" for name in self.config['class_names']))
        for i, row in enumerate(cm):
            print(f"Actual {self.config['class_names'][i][:3]}  " + "  ".join(f"{val:3d}" for val in row))

        # Calculate per-class precision, recall, and F1-score
        precision, recall, f1 = self._calculate_metrics(cm)

        print("\nPer-class Metrics:")
        print("{:12s}  {:8s}  {:8s}  {:8s}".format("Class", "Precision", "Recall", "F1-Score"))
        for i in range(self.config['num_classes']):
            print("{:12s}  {:.6f}  {:.6f}  {:.6f}".format(
                self.config['class_names'][i], precision[i], recall[i], f1[i]
            ))

        # Calculate macro average
        macro_precision = sum(precision) / len(precision)
        macro_recall = sum(recall) / len(recall)
        macro_f1 = sum(f1) / len(f1)

        print("\nMacro Average:")
        print("Precision: {:.6f}".format(macro_precision))
        print("Recall: {:.6f}".format(macro_recall))
        print("F1-Score: {:.6f}".format(macro_f1))

        return test_loss, test_accuracy, {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

    def predict(self, text):
        """
        Predict the sentiment of the given text.

        Args:
            text: Text to analyze

        Returns:
            dict: Prediction results
        """
        # Ensure model is in evaluation mode
        self.model.eval()

        # Load the best model state dictionary if available
        if os.path.exists(self.config['model_save_path']):
             # Load the state_dict and then load it into the model
            state_dict = torch.load(self.config['model_save_path'], map_location=self.device)
            self.model.load_state_dict(state_dict)
            # print(f"Model state loaded from {self.config['model_save_path']} and mapped to {self.device}") # Optional: uncomment for verbose output

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config['max_seq_length'],
            return_tensors="pt"
        )

        # Move input tensor to the selected device
        input_ids = encoding.input_ids.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids)
            probabilities = torch.softmax(outputs, dim=1)
            # Move probabilities back to CPU for numpy conversion and list conversion
            prediction = torch.argmax(probabilities.cpu(), dim=1).item()

        # Convert prediction to class label
        class_label = self.config['class_names'][prediction]

        # Extract probabilities (from CPU tensor)
        probs = probabilities[0].cpu().tolist()

        # Create result dictionary with probabilities for each class
        result = {
            'text': text,
            'prediction': prediction,
            'class': class_label,
            'probabilities': {}
        }

        # Add probabilities for each class
        for i, class_name in enumerate(self.config['class_names']):
            result['probabilities'][class_name.lower()] = probs[i]

        return result

    def plot_training_history(self):
        """Plot the training history."""
        if not self.epoch_losses:
            print("No training history to plot.")
            return

        plt.figure(figsize=(12, 5))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot validation accuracy
        if self.val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(self.val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

        print("\nTraining history plot saved as 'training_history.png'")

    def _confusion_matrix(self, y_true, y_pred, num_classes):
        """Calculate confusion matrix."""
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        return cm

    def _calculate_metrics(self, confusion_matrix):
        """Calculate precision, recall, and F1-score from confusion matrix."""
        n_classes = len(confusion_matrix)
        precision = [0] * n_classes
        recall = [0] * n_classes
        f1 = [0] * n_classes

        for i in range(n_classes):
            # True positives = confusion_matrix[i][i]
            tp = confusion_matrix[i][i]

            # Sum of i-th row = all actual samples of class i
            actual_sum = sum(confusion_matrix[i])

            # Sum of i-th column = all predicted samples of class i
            pred_sum = sum(row[i] for row in confusion_matrix)

            # Calculate metrics
            precision[i] = tp / pred_sum if pred_sum > 0 else 0
            recall[i] = tp / actual_sum if actual_sum > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        return precision, recall, f1

    def update_model_config(self, new_config):
        """
        Update model configuration parameters.

        Args:
            new_config: Dictionary with new configuration values

        Returns:
            bool: True if model was reinitialized, False otherwise
        """
        reinitialize = False

        # Check if any parameter requiring model reinitialization is changed
        model_params = ['embedding_dim', 'hidden_dim', 'num_classes', 'num_layers',
                       'dropout_rate', 'use_attention']

        for param in model_params:
            if param in new_config and param in self.config and new_config[param] != self.config[param]:
                reinitialize = True
                break

        # Update configuration
        self.config.update(new_config)

        # Reinitialize model if needed
        if reinitialize:
            # Model will be moved to the currently set device in _initialize_model
            self._initialize_model()
            print("Model reinitialized with new configuration.")

        return reinitialize


def main():
    """Main function to run the sentiment analysis system."""
    # Create the sentiment analysis system
    sentiment_system = SentimentAnalysisSystem()

    # Try to load a saved configuration
    if os.path.exists("models/sentiment_config.json"):
        sentiment_system.load_config()

    # --- Device Detection and Selection ---
    print("\n" + "="*60)
    print("Device Setup")
    print("="*60)
    print("Detecting available devices...")
    available_devices = sentiment_system.detect_available_devices()
    print(f"Available devices: {', '.join(available_devices)}")

    if len(available_devices) > 1:
        print("Please select a device for computation:")
        for i, dev in enumerate(available_devices):
            print(f"{i+1}. {dev}")
        while True:
            try:
                device_choice_idx = int(input(f"Enter device number (1-{len(available_devices)}): ")) - 1
                if 0 <= device_choice_idx < len(available_devices):
                    selected_device_name = available_devices[device_choice_idx]
                    sentiment_system.set_device(selected_device_name)
                    break # Exit the device selection loop
                else:
                    print("Invalid device number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    elif available_devices:
        # Only one device available (or just CPU), set it automatically
        sentiment_system.set_device(available_devices[0])
    else:
        # No compatible devices found (shouldn't happen as CPU is always available)
        print("No compatible devices found. Defaulting to CPU.")
        sentiment_system.set_device("cpu")

    print(f"Using device: {sentiment_system.device}")
    # --- End Device Setup ---


    while True:
        print("\n" + "="*60)
        print("Sentiment Analysis System Menu")
        print("="*60)
        print("1. Train Model")
        print("2. Test Model")
        print("3. Analyze Text")
        print("4. Configure Dataset")
        print("5. Configure Model")
        print("6. Save/Load Configuration")
        print("7. Change Device") # Added option to change device
        print("8. Exit")
        print("-"*60)

        try:
            choice = int(input("Enter your choice (1-8): ")) # Updated range

            if choice == 1:
                # Train model
                epochs = int(input("Enter number of epochs: "))
                sentiment_system.train(epochs)

            elif choice == 2:
                # Test model
                sentiment_system.test()

            elif choice == 3:
                # Analyze text
                text = input("\nEnter text to analyze: ")
                result = sentiment_system.predict(text)

                print("\nAnalysis Result:")
                print(f"Text: {result['text']}")
                print(f"Class: {result['class']}")
                print(f"Confidence:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  {class_name.capitalize()}: {prob:.4f}")

            elif choice == 4:
                # Configure dataset
                print("\nCurrent Dataset Configuration:")
                print(f"Train File: {sentiment_system.config['train_file']}")
                print(f"Validation File: {sentiment_system.config['val_file']}")
                print(f"Test File: {sentiment_system.config['test_file']}")
                print(f"Text Column: {sentiment_system.config['text_column']}")
                print(f"Label Column: {sentiment_system.config['label_column']}")
                print(f"Class Names: {sentiment_system.config['class_names']}")
                print("\nEnter new values (leave blank to keep current):")

                train_file = input(f"Train File [{sentiment_system.config['train_file']}]: ")
                val_file = input(f"Validation File [{sentiment_system.config['val_file']}]: ")
                test_file = input(f"Test File [{sentiment_system.config['test_file']}]: ")
                text_column = input(f"Text Column [{sentiment_system.config['text_column']}]: ")
                label_column = input(f"Label Column [{sentiment_system.config['label_column']}]: ")

                # Update class names if needed
                update_class_names = input("Update class names? (y/n): ").lower() == 'y'
                if update_class_names:
                    print(f"Current Class Names: {sentiment_system.config['class_names']}")
                    print("Enter new class names (comma-separated):")
                    class_names_input = input("> ")
                    if class_names_input.strip():
                        class_names = [name.strip() for name in class_names_input.split(',')]
                        sentiment_system.config['class_names'] = class_names

                # Update dataset configuration
                sentiment_system.set_dataset(
                    train_file=train_file if train_file else None,
                    val_file=val_file if val_file else None,
                    test_file=test_file if test_file else None,
                    text_column=text_column if text_column else None,
                    label_column=label_column if label_column else None
                )

            elif choice == 5:
                # Configure model parameters
                print("\nCurrent Model Configuration:")
                print(f"Embedding Dimension: {sentiment_system.config['embedding_dim']}")
                print(f"Hidden Dimension: {sentiment_system.config['hidden_dim']}")
                print(f"Number of Layers: {sentiment_system.config['num_layers']}")
                print(f"Dropout Rate: {sentiment_system.config['dropout_rate']}")
                print(f"Learning Rate: {sentiment_system.config['learning_rate']}")
                print(f"Batch Size: {sentiment_system.config['batch_size']}")
                print(f"Max Sequence Length: {sentiment_system.config['max_seq_length']}")
                print(f"Use Attention: {sentiment_system.config['use_attention']}")
                print(f"Log Interval: {sentiment_system.config['log_interval']}") # Display current log interval
                print("\nEnter new values (leave blank to keep current):")

                # Get new values
                new_config = {}

                # Get embedding dimension
                emb_dim = input(f"Embedding Dimension [{sentiment_system.config['embedding_dim']}]: ")
                if emb_dim:
                    new_config['embedding_dim'] = int(emb_dim)

                # Get hidden dimension
                hidden_dim = input(f"Hidden Dimension [{sentiment_system.config['hidden_dim']}]: ")
                if hidden_dim:
                    new_config['hidden_dim'] = int(hidden_dim)

                # Get number of layers
                num_layers = input(f"Number of Layers [{sentiment_system.config['num_layers']}]: ")
                if num_layers:
                    new_config['num_layers'] = int(num_layers)

                # Get dropout rate
                dropout_rate = input(f"Dropout Rate [{sentiment_system.config['dropout_rate']}]: ")
                if dropout_rate:
                    new_config['dropout_rate'] = float(dropout_rate)

                # Get learning rate
                learning_rate = input(f"Learning Rate [{sentiment_system.config['learning_rate']}]: ")
                if learning_rate:
                    new_config['learning_rate'] = float(learning_rate)

                # Get batch size
                batch_size = input(f"Batch Size [{sentiment_system.config['batch_size']}]: ")
                if batch_size:
                    new_config['batch_size'] = int(batch_size)

                # Get max sequence length
                max_seq_length = input(f"Max Sequence Length [{sentiment_system.config['max_seq_length']}]: ")
                if max_seq_length:
                    new_config['max_seq_length'] = int(max_seq_length)

                # Get attention usage
                use_attention = input(f"Use Attention (y/n) [{sentiment_system.config['use_attention']}]: ")
                if use_attention.lower() in ['y', 'yes', 'n', 'no']:
                    new_config['use_attention'] = use_attention.lower() in ['y', 'yes']

                # Get log interval
                log_interval = input(f"Log Interval (batches) [{sentiment_system.config['log_interval']}]: ")
                if log_interval:
                    new_config['log_interval'] = int(log_interval)

                # Update model configuration
                if new_config:
                    reinit = sentiment_system.update_model_config(new_config)
                    if reinit:
                        print("Model architecture has been updated.")
                    else:
                        print("Configuration updated without changing model architecture.")

                    # Update data loaders if batch size or max sequence length changed
                    if 'batch_size' in new_config or 'max_seq_length' in new_config:
                        sentiment_system.train_loader = sentiment_system._create_data_loader(sentiment_system.config['train_file'])
                        sentiment_system.val_loader = sentiment_system._create_data_loader(sentiment_system.config['val_file'])
                        sentiment_system.test_loader = sentiment_system._create_data_loader(sentiment_system.config['test_file'])
                        print("Data loaders have been updated.")
                else:
                    print("No changes made to configuration.")

            elif choice == 6:
                # Save/Load Configuration
                print("\nConfiguration Management:")
                print("1. Save Current Configuration")
                print("2. Load Configuration from File")
                print("3. Back to Main Menu")

                config_choice = int(input("Enter your choice (1-3): "))

                if config_choice == 1:
                    # Save configuration
                    custom_path = input(f"Save configuration to [{sentiment_system.config['config_save_path']}]: ")
                    if custom_path:
                        sentiment_system.config['config_save_path'] = custom_path
                    sentiment_system.save_config()

                elif config_choice == 2:
                    # Load configuration
                    config_path = input("Enter path to configuration file: ")
                    if not config_path:
                        config_path = sentiment_system.config['config_save_path']
                    sentiment_system.load_config(config_path)

            elif choice == 7:
                 # Change Device
                print("\n" + "="*60)
                print("Change Computation Device")
                print("="*60)
                available_devices = sentiment_system.detect_available_devices()
                print(f"Available devices: {', '.join(available_devices)}")

                if len(available_devices) > 1:
                    print("Please select a device for computation:")
                    for i, dev in enumerate(available_devices):
                        print(f"{i+1}. {dev}")
                    while True:
                        try:
                            device_choice_idx = int(input(f"Enter device number (1-{len(available_devices)}): ")) - 1
                            if 0 <= device_choice_idx < len(available_devices):
                                selected_device_name = available_devices[device_choice_idx]
                                sentiment_system.set_device(selected_device_name)
                                break # Exit the device selection loop
                            else:
                                print("Invalid device number. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                elif available_devices:
                    print("Only one device available.")
                else:
                     print("No compatible devices found.")


            elif choice == 8: # Updated exit choice
                # Exit the program
                print("Exiting the program. Goodbye!")
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 8.") # Updated range

        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optional: Add more specific error handling based on potential issues

if __name__ == "__main__":
    main()
