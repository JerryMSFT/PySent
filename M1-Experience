## Training Time Estimate for MacBook M1

The training time for the sentiment analysis model on a MacBook M1 can vary, but here's a general estimate:

- **Estimated Time**: Approximately 30 minutes to 2 hours

Factors affecting training time:

1. **Dataset Size**: The IMDB dataset used is quite large, containing 50,000 movie reviews.

2. **Model Complexity**: The LSTM model used is relatively simple, which helps keep training time reasonable.

3. **Number of Epochs**: The default setting is 5 epochs. More epochs will increase training time.

4. **Batch Size**: Larger batch sizes can speed up training but may require more memory.

5. **M1 Chip Optimization**: PyTorch has been optimized for M1 chips, which can significantly speed up training compared to older Intel Macs.

6. **Background Processes**: Other applications running on your Mac can affect training speed.

7. **Power Mode**: Ensure your MacBook is plugged in for optimal performance.

Note: The first run might take longer as it needs to download and process the IMDB dataset.

To monitor progress:
- The script prints the loss for each epoch, giving you an idea of how the training is progressing.
- You'll see output like: `Epoch [1/5], Loss: 0.6934`

If the training seems to be taking too long, you can:
1. Reduce the number of epochs in the script.
2. Use a smaller subset of the dataset for quicker experimentation.

Remember, this is a one-time process. Once trained, the model can analyze sentiments quickly.
