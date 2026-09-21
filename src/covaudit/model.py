"""The base model. Deliberately ordinary: it is not the contribution."""
from sklearn.ensemble import HistGradientBoostingClassifier


def train_model(X_train, y_train, seed=0):
    """Train a gradient boosting classifier on the training pile and return it."""
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model
