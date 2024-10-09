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

## Installation and Setup

Choose the instructions for your operating system:

### MacBook (Intel)

1. Ensure you have Python 3.6+ installed. Check your version:
   ```
   python3 --version
   ```
   If needed, download Python from [python.org](https://www.python.org/downloads/).

2. Create a virtual environment:
   ```
   python3 -m venv sentiment_env
   ```

3. Activate the environment:
   ```
   source sentiment_env/bin/activate
   ```

4. Install required packages:
   ```
   pip3 install torch torchvision torchaudio torchtext torchdata
   ```

### MacBook (M1/Apple Silicon)

1. Install Miniconda:
   - Download the Miniconda installer for Mac M1 from [here](https://docs.conda.io/en/latest/miniconda.html#macos-installers).
   - Choose the Apple M1 version.
   - In Terminal, run:
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

4. Install required packages:
   ```
   conda install pytorch torchvision torchaudio -c pytorch
   pip install torchtext torchdata
   ```

### Windows

1. Ensure you have Python 3.6+ installed. Check your version:
   ```
   python --version
   ```
   If needed, download Python from [python.org](https://www.python.org/downloads/).

2. Open Command Prompt and create a virtual environment:
   ```
   python -m venv sentiment_env
   ```

3. Activate the environment:
   ```
   sentiment_env\Scripts\activate
   ```

4. Install required packages:
   ```
   pip install torch torchvision torchaudio torchtext torchdata
   ```

### Linux

1. Ensure you have Python 3.6+ installed. Check your version:
   ```
   python3 --version
   ```
   If needed, install Python using your distribution's package manager.

2. Create a virtual environment:
   ```
   python3 -m venv sentiment_env
   ```

3. Activate the environment:
   ```
   source sentiment_env/bin/activate
   ```

4. Install required packages:
   ```
   pip3 install torch torchvision torchaudio torchtext torchdata
   ```

## Usage

1. Clone this repository or download the `sentiment.py` file.

2. Ensure you're in the correct environment:
   - For MacBook M1: `conda activate sentiment`
   - For others: `source sentiment_env/bin/activate` (use `sentiment_env\Scripts\activate` on Windows)

3. Run the script:
   ```
   python sentiment.py
   ```

4. The script will:
   - Download and prepare the IMDB dataset (first run only)
   - Train the sentiment analysis model
   - Enter an interactive mode for testing

5. In the interactive mode, enter your own text to analyze its sentiment.

6. When done, deactivate the environment:
   - For MacBook M1: `conda deactivate`
   - For others: `deactivate`

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

## Troubleshooting

- If you encounter package installation issues, ensure you're using the latest pip version: `pip install --upgrade pip`
- For GPU support, refer to PyTorch's official documentation for specific installation instructions.
- If you face "out of memory" errors, try reducing the batch size in the script.

## Future Improvements

- Implement cross-validation for more robust evaluation
- Experiment with more advanced architectures (e.g., bidirectional LSTM, attention mechanisms)
- Add functionality to save and load trained models

## Contributing

Feel free to fork this project and submit pull requests with improvements or open issues for any bugs or feature requests.

## License

This project is open-source and available under the MIT License.

# Using the Model

## Example Inputs for Sentiment Analysis

After training, you can input various types of text to analyze their sentiment. Here are some examples to try:

1. Movie Reviews:
   - "This film was a masterpiece of storytelling and visual effects."
   - "I found the movie boring and predictable, with poor acting throughout."
   - "While the cinematography was stunning, the plot left much to be desired."

2. Product Reviews:
   - "This gadget exceeded my expectations. It's user-friendly and efficient."
   - "The product arrived damaged and customer service was unhelpful."
   - "For the price, it's a decent option, but there are better alternatives available."

3. Restaurant Experiences:
   - "The food was delicious and the service was impeccable."
   - "Our dinner was ruined by slow service and cold, unappetizing dishes."
   - "The ambiance was nice, but the menu was limited and overpriced."

4. Book Opinions:
   - "I couldn't put this book down! The characters were so well-developed."
   - "The story started strong but fell apart in the second half."
   - "While not groundbreaking, it was an enjoyable and easy read."

5. General Statements:
   - "I'm having a wonderful day and everything is going great!"
   - "This has been the worst week of my life."
   - "The weather is a bit gloomy, but I'm looking forward to the weekend."

When you input these sentences, the model will analyze them and provide a sentiment score. Remember:
- Scores closer to 1 indicate more positive sentiment.
- Scores closer to 0 indicate more negative sentiment.
- Scores around 0.5 may indicate neutral or mixed sentiment.

Try these examples and see how well the model captures the sentiment in each statement. You can also create your own sentences to test the model's capabilities and limitations.

## Expressing Movie Sentiments for Analysis

You can provide sentiment about a movie by simply typing a statement or opinion about it. The model will analyze the text you enter and determine whether it expresses a positive or negative sentiment. Here are some examples:

1. Simple statements:
   - "I loved the movie TRON."
   - "TRON was a terrible film."
   - "The new superhero movie was amazing!"

2. More detailed opinions:
   - "TRON had incredible visual effects but the story was confusing."
   - "I enjoyed TRON for its nostalgic value, even though the plot was weak."
   - "The acting in TRON was mediocre, but the concept was innovative."

3. Comparisons or context:
   - "TRON was much better than I expected."
   - "Compared to the original, the TRON sequel was disappointing."
   - "For a sci-fi movie from the 80s, TRON was groundbreaking."

How to use:
1. After the model is trained, you'll see a prompt: "Enter text:"
2. Type your statement about the movie and press Enter.
3. The model will analyze your input and provide a sentiment score.

Example interaction:

```
Enter text: I loved the movie TRON.
Sentiment: Positive (Score: 0.89)

Enter text: TRON was visually stunning but the plot was hard to follow.
Sentiment: Neutral (Score: 0.52)

Enter text: The sequel to TRON was a huge letdown.
Sentiment: Negative (Score: 0.21)
```

Remember:
- Be clear and direct in your statements.
- You can mention specific aspects of the movie (plot, acting, effects, etc.).
- The model analyzes the overall sentiment of your entire statement.
- You don't need to use any special format or keywords; just express your opinion naturally.
