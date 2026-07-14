import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: Generate a synthetic binary classification dataset
# ============================================================
# X -> features, y -> binary labels (0 or 1)
X, y = make_classification(
    n_samples=300,      # total number of data points
    n_features=2,        # number of input features (2D so we can visualize)
    n_redundant=0,        # no redundant (linearly dependent) features
    n_informative=2,       # both features carry useful information
    n_clusters_per_class=1, # each class forms a single cluster
    random_state=42         # for reproducibility
)

# ============================================================
# STEP 2: Split into training and testing sets
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# STEP 3: Feature scaling (important for gradient descent to converge well)
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train, transform train
X_test = scaler.transform(X_test)         # only transform test (no fit!)


# ============================================================
# STEP 4: Define the Sigmoid function
# ============================================================
def sigmoid(z):
    """
    Sigmoid activation function.
    Converts any real number into a value between 0 and 1.
    Formula: sigmoid(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))


# ============================================================
# STEP 5: Define the Logistic Regression class
# ============================================================
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        """
        Constructor to initialize hyperparameters.
        learning_rate -> step size for gradient descent
        n_iterations  -> number of times we update weights
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None   # will hold feature weights (coefficients)
        self.bias = None      # will hold the bias/intercept term
        self.loss_history = []  # to track loss over iterations (for plotting)

    def fit(self, X, y):
        """
        Train the logistic regression model using Gradient Descent.
        X -> training features, shape (n_samples, n_features)
        y -> training labels, shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero (common starting point)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # ---- Gradient Descent Loop ----
        for i in range(self.n_iterations):

            # 1. Compute the linear combination: z = X.w + b
            linear_model = np.dot(X, self.weights) + self.bias

            # 2. Apply sigmoid to get predicted probabilities (0 to 1)
            y_predicted = sigmoid(linear_model)

            # 3. Compute gradients of the loss w.r.t weights and bias
            #    These formulas come from differentiating the
            #    binary cross-entropy (log loss) function.
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Update weights and bias in the direction that reduces loss
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 5. Compute and store the loss (log loss / binary cross-entropy)
            #    Small epsilon (1e-9) added to avoid log(0) errors
            loss = -np.mean(
                y * np.log(y_predicted + 1e-9) +
                (1 - y) * np.log(1 - y_predicted + 1e-9)
            )
            self.loss_history.append(loss)

    def predict_proba(self, X):
        """
        Returns the predicted probability of belonging to class 1.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Converts probabilities into class labels (0 or 1)
        using the given threshold (default 0.5).
        """
        y_probabilities = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in y_probabilities])


# ============================================================
# STEP 6: Train the model
# ============================================================
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# ============================================================
# STEP 7: Make predictions on the test set
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# STEP 8: Evaluate the model
# ============================================================
def accuracy(y_true, y_pred):
    """Calculates simple classification accuracy."""
    return np.mean(y_true == y_pred)

print("Learned Weights:", model.weights)
print("Learned Bias:", model.bias)
print("Test Accuracy:", accuracy(y_test, y_pred))

# ============================================================
# STEP 9: Plot the loss curve (to see model converging over time)
# ============================================================
plt.figure(figsize=(6, 4))
plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Loss Curve During Training")
plt.show()