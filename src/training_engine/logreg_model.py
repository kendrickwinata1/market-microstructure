import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from base_model import BaseModel

class LogRegModel(BaseModel):
    """
    Logistic Regression model (with Elastic Net regularization) built on top of BaseModel.
    """

    def __init__(self, file_path):
        super().__init__(file_path)
        # Elastic Net parameters
        l1_ratio = 0.5  # L1 weight (0=L2, 1=L1)
        alpha = 1.0     # Regularization strength (higher=stronger)
        self.model = LogisticRegression(
            class_weight="balanced",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            C=1 / alpha,
            solver="saga",
            max_iter=1000,
        )

    def train(self):
        """
        Train logistic regression model using the training split from BaseModel.
        Saves model to outputs/logistic_regression_model_updated.pkl.
        """
        self.train_test_split_time_series()

        print("y_train value counts: ", self.y_train.value_counts())
        print("X_train shape: ", self.X_train.shape)
        print("y_train shape: ", self.y_train.shape)

        self.model.fit(self.X_train, self.y_train)

        # Save the model to a file
        joblib.dump(self.model, "outputs/logistic_regression_model_updated.pkl")
        print("Model saved to outputs/logistic_regression_model_updated.pkl")

    def predict(self):
        """
        Predict using the trained model on the test split.
        Returns predicted class labels for X_test.
        """
        print("X_test shape: ", self.X_test.shape)
        predicted_categories = self.model.predict(self.X_test)
        return predicted_categories

    def evaluate(self, X, y):
        """
        Placeholder for evaluation logic.
        To be implemented if needed.
        """
        pass
