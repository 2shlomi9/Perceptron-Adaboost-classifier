import numpy as np

class Perceptron:
    def __init__(self):
        self.w = None
        self.count_mistakes = 0

    def fit(self, data_1, data_2):
        """
        Train the perceptron algorithm on the given data.
        """
        y1 = np.ones(data_1.shape[0])  # Labels for class 1
        y2 = -np.ones(data_2.shape[0])  # Labels for class -1

        X = np.vstack((data_1, data_2))  # Merge data
        Y = np.concatenate((y1, y2))  # Merge labels

        self.w = np.zeros(data_1.shape[1])  # Initialize weight vector
        self.count_mistakes = 0

        mistakes = True  # Flag to check if mistakes were made in an epoch

        while mistakes:
            mistakes = False  # Reset flag at start of each epoch

            for i in range(len(X)):
                prediction = np.dot(self.w, X[i])  # Compute prediction

                if prediction <= 0 and Y[i] == 1:  # Misclassified positive example
                    self.w += X[i]  # Update weights
                    mistakes = True
                    self.count_mistakes += 1
                elif prediction > 0 and Y[i] == -1:  # Misclassified negative example
                    self.w -= X[i]  # Update weights
                    mistakes = True
                    self.count_mistakes += 1

            if not mistakes:  # If no mistakes were made, stop training
                break

    def predict(self, X):
        """
        Predict the label of new data points.
        """
        return np.sign(np.dot(X, self.w))

    def get_weights(self):
        """
        Return the weight vector.
        """
        return self.w

    def get_mistake_count(self):
        """
        Return the number of mistakes made during training.
        """
        return self.count_mistakes
