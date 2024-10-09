# Sentiment Analysis with LSTM

## Overview

This project implements a sentiment analysis model using a Long Short-Term Memory (LSTM) neural network. It's designed to classify movie reviews as either positive or negative based on the text content. The model is trained on the IMDB dataset, which contains a large collection of movie reviews.

## Features

- LSTM-based neural network for sentiment classification
- Utilizes PyTorch for model implementation
- Trains on the IMDB dataset
- Includes an interactive mode for real-time sentiment analysis of user input

## Requirements

- Python 3.6+
- PyTorch
- torchtext
- torchdata

## Installation

### For Mac M1 Users (using Miniconda)

1. Install Miniconda:
   - Download the Miniconda installer for Mac M1 from [here](https://docs.conda.io/en/latest/miniconda.html#macos-installers).
   - Choose the Apple M1 version.
   - Open Terminal and navigate to the directory where you downloaded the installer.
   - Run the following command to install Miniconda:
     ```
     sh Miniconda3-latest-MacOSX-arm64.sh
     ```
   - Follow the prompts to complete the installation.

2. Create a new conda environment:
   ```
   conda create -n sentiment python=3.9
   ```

3. Activate the environment:
   ```
   conda activate sentiment
   ```

4. Install PyTorch and related packages:
   ```
   conda install pytorch torchvision torchaudio -c pytorch
   pip install torchtext torchdata
   ```

### For Other Users

1. Clone this repository or download the `sentiment.py` file.

2. Install the required packages:
   ```
   pip install torch torchtext torchdata
   ```

## Usage

1. Ensure you're in the correct conda environment (for M1 Mac users):
   ```
   conda activate sentiment
   ```

2. Run the script:
   ```
   python sentiment.py
   ```

3. The script will:
   - Download and prepare the IMDB dataset (first run only)
   - Train the sentiment analysis model
   - Enter an interactive mode for testing

4. In the interactive mode, you can enter your own text to analyze its sentiment.

## How it Works

1. **Data Preparation**: The IMDB dataset is loaded and processed. Reviews are tokenized and converted to numerical sequences.

2. **Model Architecture**: 
   - Embedding Layer: Converts word indices to dense vectors
   - LSTM Layer: Processes the sequence of word embeddings
   - Fully Connected Layer: Produces the final sentiment score
   - Sigmoid Activation: Squashes the output to a range of 0 to 1

3. **Training**: The model is trained for 5 epochs on the IMDB dataset.

4. **Prediction**: The trained model predicts sentiment scores for new text inputs.

## Customization

- Adjust model parameters in the script (e.g., `EMBEDDING_DIM`, `HIDDEN_DIM`, `NUM_EPOCHS`) to experiment with different configurations.
- Modify the `SentimentLSTM` class to try different neural network architectures.

## Limitations

- The model is trained on movie reviews and may not perform as well on other types of text.
- As a simple LSTM model, it may not capture extremely complex language patterns or context.

## Future Improvements

- Implement cross-validation for more robust evaluation
- Experiment with more advanced architectures (e.g., bidirectional LSTM, attention mechanisms)
- Add functionality to save and load trained models

## Troubleshooting

- If you encounter any issues with package compatibility, try creating a new conda environment with Python 3.9 as shown in the installation instructions.
- For M1 Mac users, ensure you're using the ARM64 versions of the packages.

## Contributing

Feel free to fork this project and submit pull requests with improvements or open issues for any bugs or feature requests.

## License

This project is open-source and available under the MIT License.



This updated README now includes:

1. Specific instructions for Mac M1 users using Miniconda.
2. Steps to create and activate a conda environment.
3. Instructions for installing PyTorch and related packages in the conda environment.
4. A reminder to activate the correct environment before running the script.
5. A troubleshooting section addressing potential issues specific to M1 Macs.

These additions will help Mac M1 users set up the necessary environment to run your sentiment analysis script without compatibility issues. The instructions use Python 3.9, which is generally well-supported for most machine learning libraries on M1 Macs.

To use this updated README:

1. Replace the content of your existing `README.md` file with this new version.
2. If you don't have a README file yet, create a new file named `README.md` in the same directory as your `sentiment.py` script and paste this content into it.

This comprehensive README should now cater to a wider range of users, including those with M1 Macs, making your project more accessible and easier to set up.
