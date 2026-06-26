import numpy as np

class GeneralizedLinearModel:
    """
    A base class for linear models optimized via Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _activate(self, z):
        """Placeholder for activation function. Overridden by child classes."""
        raise NotImplementedError

    def _compute_cost(self, y_true, y_pred, n_samples):
        """Placeholder for cost function. Overridden by child classes."""
        raise NotImplementedError

    def fit(self, X, y):
        """
        Shared training loop using Gradient Descent.
        """
        n_samples, n_features = X.shape
        
        # Shared Initialization
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        # Shared Gradient Descent Optimization Loop
        for _ in range(self.n_iterations):
            linear_combination = np.dot(X, self.weights) + self.bias
            y_predicted = self._activate(linear_combination)
            
            # Compute specific cost
            cost = self._compute_cost(y, y_predicted, n_samples)
            self.cost_history.append(cost)

            # The exact same gradient formula applies to both models!
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Shared parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_raw(self, X):
        """Returns the continuous prediction (raw values or probabilities)"""
        linear_combination = np.dot(X, self.weights) + self.bias
        return self._activate(linear_combination)


# =====================================================================
# Child Class 1: Linear Regression
# =====================================================================
class LinearRegression(GeneralizedLinearModel):
    def _activate(self, z):
        return z  # Identity function for regression

    def _compute_cost(self, y_true, y_pred, n_samples):
        return (1 / (2 * n_samples)) * np.sum((y_pred - y_true)**2)

    def predict(self, X):
        return self.predict_raw(X)


# =====================================================================
# Child Class 2: Logistic Regression
# =====================================================================
class LogisticRegression(GeneralizedLinearModel):
    def _activate(self, z):
        z = np.clip(z, -500, 500)  # Numerical stability check
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y_true, y_pred, n_samples):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (1 / n_samples) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_raw(X)
        return np.where(probabilities >= threshold, 1, 0)


# =====================================================================
# Verification Execution
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression

    print("--- Testing Shared Linear Regression ---")
    X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    reg_model = LinearRegression(learning_rate=0.1, n_iterations=300)
    reg_model.fit(X_reg, y_reg)
    print(f"Final Regressor Loss (MSE): {reg_model.cost_history[-1]:.4f}")
    print(f"Learned Coefficients: {reg_model.weights}\n")

    print("--- Testing Shared Logistic Regression ---")
    X_clf, y_clf = make_classification(n_samples=100, n_features=2, random_state=42)
    clf_model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    clf_model.fit(X_clf, y_clf)
    print(f"Final Classifier Loss (BCE): {clf_model.cost_history[-1]:.4f}")
    acc = np.mean(clf_model.predict(X_clf) == y_clf)
    print(f"Classifier Training Accuracy: {acc * 100:.2f}%")
