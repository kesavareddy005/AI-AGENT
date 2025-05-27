# Comprehensive Machine Learning and Deep Learning Projects

Welcome to this comprehensive repository featuring three distinct yet insightful Jupyter notebooks:

1. **IMDB Sentiment Analysis**
2. **Health Care Status Prediction**
3. **Cat vs Dog Image Classification**

This documentation provides an in-depth overview of each project, complete with project content, key technologies, descriptions, code highlights, outputs, and future directions.

---

## 📁 1. Project Content

### IMDB Sentiment Analysis (`IMDB (1).ipynb`)
A text classification project that predicts the sentiment (positive or negative) of movie reviews from the IMDB dataset. It demonstrates natural language processing and deep learning techniques.

### Health Care Status Prediction (`HEALTH_CARE.ipynb`)
A classification task that leverages PyTorch to predict the health status of individuals based on various medical parameters.

### Cat vs Dog Image Classification (`cat_dog (1).ipynb`)
A computer vision project that classifies images of cats and dogs using convolutional neural networks (CNNs) with PyTorch.

---

## 🔧 2. Key Technologies

✅ **Python 3**  
✅ **Jupyter Notebook**  
✅ **Pandas & NumPy**  
✅ **Matplotlib & Seaborn**  
✅ **scikit-learn**  
✅ **PyTorch**  
✅ **Torchvision**  
✅ **Google Colab / Local GPU**

---

## 📝 3. Description

### IMDB Sentiment Analysis
This notebook begins with loading and preprocessing the IMDB dataset, transforming text data into numerical representations using tokenization and embedding layers. The project applies a recurrent neural network (RNN) or LSTM to capture the temporal dependencies of words in a review. It culminates with evaluation metrics like accuracy, precision, recall, and F1-score to measure performance.

Key steps:
- Text preprocessing (lowercasing, punctuation removal)
- Tokenization and word embeddings
- Model architecture (LSTM/GRU-based)
- Model training and validation
- Visualization of loss and accuracy

---

### Health Care Status Prediction
The notebook implements a classification model to predict an individual's health status based on numeric features (e.g., cholesterol level, blood pressure, etc.). It includes:

- Data exploration and visualization
- Preprocessing (scaling, normalization)
- Model architecture: Multi-Layer Perceptron (MLP)
- Training and evaluation
- Classification report and confusion matrix

---

### Cat vs Dog Image Classification
This notebook leverages transfer learning for efficient and accurate image classification.

- Dataset loading using `ImageFolder`
- Data augmentation and preprocessing
- Model architecture: Pre-trained CNN (ResNet / VGG)
- Model fine-tuning for cat-dog images
- Training loop and early stopping
- Visualization of predictions on test images

---

## 💻 4. Project Code

### IMDB (1).ipynb
```python
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# ... data preprocessing code
# Model architecture
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.rnn(x)
        return self.fc(hidden[-1])
