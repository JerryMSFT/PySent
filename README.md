# IMDB Sentiment Analysis with BERT

This project implements a sentiment analysis model using a BERT-based transformer from the Hugging Face library. The model is trained on the IMDB dataset and can predict sentiment (positive or negative) for given text inputs.

## Features

- Uses BERT (bert-base-uncased) for sentiment classification
- Trains on the IMDB dataset
- Provides an interactive mode for real-time sentiment analysis
- Utilizes PyTorch and the Transformers library

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets

You can install the required packages using pip:

```
pip install torch transformers datasets
```

## Usage

1. Run the script:

```
python sentiment_analysis.py
```

2. The script will perform the following steps:
   - Load the IMDB dataset
   - Initialize the BERT model and tokenizer
   - Train the model on the training set
   - Evaluate the model on the test set
   - Enter interactive mode

3. In interactive mode, you can input sentences to analyze their sentiment:
   - Type a sentence and press Enter
   - The model will output the predicted sentiment (Positive or Negative) and a score
   - Type 'quit' to exit the interactive mode

## Example

```
Enter text: This movie was absolutely fantastic!
Sentiment: Positive (Score: 0.98)

Enter text: I was disappointed by the poor acting and weak plot.
Sentiment: Negative (Score: 0.03)

Enter text: quit
Thank you for using the sentiment analyzer!
```

## Customization

You can modify the following parameters in the script to customize the model:

- `num_epochs`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `max_length` in `IMDBDataset`: Maximum sequence length for tokenization

## Notes

- The model uses the GPU if available, otherwise it falls back to CPU.
- Training may take some time depending on your hardware.
- The interactive mode allows for quick testing of the model's performance on custom inputs.

## Future Improvements

- Implement early stopping to prevent overfitting
- Add support for saving and loading trained models
- Experiment with different pre-trained models or fine-tuning strategies

Feel free to contribute to this project or report any issues you encounter!