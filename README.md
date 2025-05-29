# Disaster Tweet Classification

This project is a solution to the [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started) *"Natural Language Processing with Disaster Tweets"*, where the goal is to classify whether a tweet is about a real disaster or not.

## Overview

We developed a binary classification model using a Bidirectional LSTM neural network. The workflow includes data preprocessing, handling class imbalance, text tokenization, hyperparameter tuning with Keras Tuner, and generating a submission file.

## Libraries Used

- `pandas`, `numpy`: Data handling
- `matplotlib`, `seaborn`: Visualization
- `re`, `os`: Utility functions
- `scikit-learn`: Feature extraction and class weights
- `tensorflow` and `keras`: Deep learning framework
- `keras_tuner`: Hyperparameter optimization

## Model Architecture

The model is a Keras `Sequential` model consisting of:

- **Embedding Layer**: Converts words into dense vectors of fixed size.
- **Bidirectional LSTM**: Processes input in both forward and backward directions.
- **Dense Layer**: ReLU activated hidden layer.
- **Dropout Layers**: Prevent overfitting.
- **Output Layer**: Sigmoid-activated for binary classification.

## Hyperparameter Tuning

Hyperparameters were tuned using `keras_tuner.RandomSearch`. The following were optimized:

- Embedding dimension: `[32, 64, 128]`
- LSTM units: `[32, 64, 128]`
- Dropout and recurrent dropout: `0.1` to `0.5`
- Dense layer units: `[16, 32, 64]`
- Dense layer dropout: `0.1` to `0.5`
- Optimizer: `adam`, `rmsprop`, `nadam`
- Learning rate: `[1e-2, 1e-3, 1e-4]`

### Best Configuration (Trial #0)
```python
{
  'embedding_dim': 32,
  'lstm_units': 32,
  'dropout': 0.1,
  'recurrent_dropout': 0.3,
  'dense_units': 64,
  'dense_dropout': 0.4,
  'optimizer': 'rmsprop',
  'learning_rate': 0.001
}
