# HFT-Time-Cyclic-Improvement

Forecasting cryptocurrency prices and market trends using high-frequency data is inherently challenging due to the significant noise and complex dynamics within market microstructures. Traditional deep learning approaches relying solely on standard market data such as Open, High, Low, Close, Volume (OHLCV), or Limit Order Book (LOB) data frequently experience issues like overfitting and poor generalization on unseen market conditions.

This project investigates whether integrating cyclical temporal features—specifically, minute-of-hour, hour-of-day, and day-of-week—can enhance the predictive capabilities of deep learning models.

We conducted comprehensive experiments using high-frequency Ethereum trading data and evaluated the effectiveness of incorporating these cyclical temporal features into three distinct deep learning architectures:

- **Long Short-Term Memory (LSTM)**

- **Convolutional Neural Networks (CNN)**

- **Hybrid CNN-LSTM**

The evaluation encompasses two core predictive tasks:

- **Regression Task**: Minute-ahead price prediction

- **Classification Task**: Short-term market trend prediction\\