import pricePrediction
import trendPrediction


def main():
    # Train and evaluate models for price prediction
    for i in range(pricePrediction.NUM_MODELS):
        for j in range(pricePrediction.NUM_FEATURE_SETS):
            pricePrediction.pricePrediction(pricePrediction.MODEL_NAMES[i], i, j)

    # Train and evaluate models for trend prediction
    for i in range(trendPrediction.NUM_MODELS):
        for j in range(trendPrediction.NUM_FEATURE_SETS):
            trendPrediction.trendPrediction(trendPrediction.MODEL_NAMES[i], i, j)

if __name__ == "__main__":
    main()