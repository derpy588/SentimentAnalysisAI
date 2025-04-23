import torch
from torch import nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on important parts of the sequence.
    Computes a weighted sum of the LSTM hidden states based on their importance.
    """
    def __init__(self, hidden_dim):
        """
        Init the attention layer.
        
        Args:
            hidden_dim: Dimension of the hidden states
        """
        super(AttentionLayer, self).__init__()
        # Attention parameters
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_output):
        """
        Apply attention mechanism to LSTM output sequence.
        
        Args:
            lstm_output: Output from the LSTM [batch_size, seq_len, hidden_dim]
            
        Returns:
            context: Weighted sum of hidden states [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        # tanh(Wh) shape: [batch_size, seq_len, hidden_dim]
        tanh_output = torch.tanh(self.attention(lstm_output))
        
        # Calculate scalar attention weights
        # attention_weights shape: [batch_size, seq_len, 1]
        attention_weights = self.context_vector(tanh_output)
        
        # Normalize attention weights
        # attention_weights shape: [batch_size, seq_len]
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        
        # Calculate context vector as weighted sum of LSTM outputs
        # context shape: [batch_size, hidden_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, attention_weights

class SentimentClassifier(nn.Module):
    """
    A neural network for sentiment classification with customizable classes.
    
    Architecture:
    - Embedding layer converts token IDs to dense vectors
    - Bidirectional LSTM processes the sequence of embeddings
    - Attention mechanism focuses on important parts of the sequence
    - Fully connected layers map to classification logits
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=3, 
                 num_layers=2, dropout_rate=0.3, use_attention=True):
        """
        Initialize the sentiment classifier model.
        
        Args:
            vocab_size: Size of the vocabulary from the tokenizer
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_classes: Number of sentiment classes (default: 3 for neg/neutral/pos)
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability for regularization
            use_attention: Whether to use attention mechanism
        """
        super(SentimentClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  # Half the hidden size for bidirectional
            num_layers=num_layers, 
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim) if use_attention else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        
        # Get batch size and sequence length
        batch_size, seq_length = x.size()
        
        # Apply embedding and dropout
        # embedded shape: [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(self.embedding(x))
        
        # Pass through bidirectional LSTM
        # lstm_output shape: [batch_size, seq_length, hidden_dim]
        # hidden shape: [2*num_layers, batch_size, hidden_dim//2]
        lstm_output, (hidden, _) = self.lstm(embedded)
        
        if self.use_attention:
            # Apply attention mechanism
            # context shape: [batch_size, hidden_dim]
            context, _ = self.attention(lstm_output)
            
            # Apply first fully connected layer
            # x shape: [batch_size, hidden_dim//2]
            x = F.relu(self.fc1(context))
        else:
            # Concatenate the final hidden states from both directions
            # hidden shape: [2*num_layers, batch_size, hidden_dim//2]
            # We want the last layer's hidden states from both directions
            hidden_forward = hidden[2*self.num_layers-2]  # [batch_size, hidden_dim//2]
            hidden_backward = hidden[2*self.num_layers-1]  # [batch_size, hidden_dim//2]
            
            # Concatenate the forward and backward final hidden states
            # hidden_concat shape: [batch_size, hidden_dim]
            hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
            
            # Apply dropout
            hidden_concat = self.dropout(hidden_concat)
            
            # Apply first fully connected layer
            x = F.relu(self.fc1(hidden_concat))
        
        # Apply dropout and second fully connected layer
        x = self.dropout(x)
        # logits shape: [batch_size, num_classes]
        logits = self.fc2(x)
        
        return logits