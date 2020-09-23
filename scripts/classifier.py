import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from scripts.config import PREDICT_SCRIPT_PATH, TRAIN_SCRIPT_PATH, MODEL_PATH, TRAIN_DATA_PATH, PREDICT_DATA_PATH
from scripts.data_reader import CSVReader


class XgbClassifier():
    seed = 7
    test_size = 0.33

    @staticmethod
    def preprocess_dataset(usecase: str, **kwargs):
        xgb = XgbClassifier()
        if usecase == "predict":
            print(f"Reading file : {PREDICT_SCRIPT_PATH}")
            dataset = xgb.__get_dataset(PREDICT_SCRIPT_PATH)
        elif usecase == "train":
            print(f"Reading file : {TRAIN_SCRIPT_PATH}")
            dataset = xgb.__get_dataset(TRAIN_SCRIPT_PATH)

        if dataset is None:
            return None, None

        dataset['variety'] = dataset['variety'].replace(['Setosa'], '0')
        dataset['variety'] = dataset['variety'].replace(['Versicolor'], '1')
        dataset['variety'] = dataset['variety'].replace(['Virginica'], '2')
        print(f"Completed changing labels...")

        if usecase == "predict":
            print(f"Saving file : {PREDICT_DATA_PATH}")
            dataset.to_csv(PREDICT_DATA_PATH, index=False)
        elif usecase == "train":
            print(f"Saving file : {TRAIN_DATA_PATH}")
            dataset.to_csv(TRAIN_DATA_PATH, index=False)


    @staticmethod
    def predict_model(**kwargs):
        xgb = XgbClassifier()
        print(f"Reading file : {PREDICT_DATA_PATH}")
        dataset = xgb.__get_dataset(PREDICT_DATA_PATH)

        print(f"Extracting features...")
        features, _ = xgb.__split_features_and_labels(dataset)
        model = xgb.__load_latest_model()

        print(f"Running model...")
        predictions = model.predict(features)
        print(f"Predictions : {predictions}")
        return predictions

    @staticmethod
    def train_model(**kwargs):
        xgb = XgbClassifier()
        print(f"Reading file : {TRAIN_DATA_PATH}")
        dataset = xgb.__get_dataset(TRAIN_DATA_PATH)

        print(f"Extracting features and labels...")
        features, labels = xgb.__split_features_and_labels(dataset)

        print(f"Splits train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=xgb.test_size,
                                                            random_state=xgb.seed)

        print(f"Training model..")
        model = XGBClassifier()
        model.fit(X_train, y_train)

        print(f"Saving model..")
        xgb.__save_model(model)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy  {accuracy * 100}%')

    def __get_dataset(self, filepath):
        return CSVReader().read(filepath)

    def __save_model(self, model):
        pickle.dump(model, open(MODEL_PATH, "wb"))

    def __load_latest_model(self):
        model = pickle.load(open(MODEL_PATH, "rb"))
        return model

    def __split_features_and_labels(self, dataset):
        if dataset is None:
            return None, None

        arr = dataset.values
        features = arr[:, 0:4]
        labels = arr[:, 4]
        return features, labels


def main():
    XgbClassifier.train_model()
    XgbClassifier.predict_model()


if __name__ == "__main__":
    main()
