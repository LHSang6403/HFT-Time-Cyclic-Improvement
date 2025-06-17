import price_prediction
import trend_prediction


def main():
    # Train and evaluate models for price prediction
    for i in range(price_prediction.NUM_MODELS):
        for j in range(price_prediction.NUM_FEATURE_SETS):
            price_prediction.pricePrediction(price_prediction.MODEL_NAMES[i], i, j)

    # Train and evaluate models for trend prediction
    for i in range(trend_prediction.NUM_MODELS):
        for j in range(trend_prediction.NUM_FEATURE_SETS):
            trend_prediction.trendPrediction(trend_prediction.MODEL_NAMES[i], i, j)

if __name__ == "__main__":
    main()