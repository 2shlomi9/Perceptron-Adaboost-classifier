import numpy as np
import random

class Adaboost:
    def __init__(self, k=8):
        self.k = k
        self.hypotheses = []
        self.alphas = []

    def generate_hypothesis_set(self, data):
        """
        Generate a set of weak classifiers (lines) defined by every pair of points.
        """
        hypotheses = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                p1, p2 = data[i], data[j]

                def weak_classifier(x, p1=p1, p2=p2):
                    """
                    Classify based on the position relative to the line defined by p1 and p2.
                    """
                    return 1 if (x[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (x[0] - p1[0]) else -1

                hypotheses.append(weak_classifier)
        return hypotheses

    def fit(self, data_1, data_2):
        """
        Train AdaBoost classifier.
        """
        # Combine data and assign labels
        data = np.vstack([data_1, data_2])
        labels = np.hstack([np.ones(len(data_1)), -np.ones(len(data_2))])

        # Split into train and test
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        train_indices = indices[:len(data) // 2]
        test_indices = indices[len(data) // 2:]

        self.train_data, self.train_labels = data[train_indices], labels[train_indices]
        self.test_data, self.test_labels = data[test_indices], labels[test_indices]

        # Generate hypothesis set
        self.hypotheses = self.generate_hypothesis_set(self.train_data)

        # Initialize weights
        n_train = len(self.train_data)
        D = np.ones(n_train) / n_train

        selected_hypotheses = []
        self.alphas = []

        for t in range(self.k):
            # Calculate weighted error for each hypothesis
            errors = []
            for h in self.hypotheses:
                predictions = np.array([h(x) for x in self.train_data])
                weighted_error = np.sum(D * np.where(predictions != self.train_labels, 1, 0))
                errors.append(weighted_error)

            # Select the hypothesis with minimum error
            best_h_idx = np.argmin(errors)
            best_h = self.hypotheses[best_h_idx]
            min_error = errors[best_h_idx]

            # Compute alpha
            alpha_t = 0.5 * np.log((1 - min_error) / max(min_error, 1e-10))
            self.alphas.append(alpha_t)
            selected_hypotheses.append(best_h)

            # Update weights
            predictions = np.array([best_h(x) for x in self.train_data])
            D *= np.exp(-alpha_t * self.train_labels * predictions)
            D /= np.sum(D)  # Normalize

        self.selected_hypotheses = selected_hypotheses

    def predict(self, x, t=None):
        """
        Predict label for a given x using the first t weak classifiers.
        If t is None, use all trained classifiers.
        """
        if t is None:
            t = self.k
        return np.sign(sum(self.alphas[i] * self.selected_hypotheses[i](x) for i in range(t)))

    def compute_errors(self):
        """
        Compute empirical and true errors for each iteration.
        """
        empirical_errors = []
        true_errors = []

        for t in range(1, self.k + 1):
            emp_error = np.mean([self.predict(x, t) != y for x, y in zip(self.train_data, self.train_labels)])
            true_error = np.mean([self.predict(x, t) != y for x, y in zip(self.test_data, self.test_labels)])

            empirical_errors.append(emp_error)
            true_errors.append(true_error)

        return empirical_errors, true_errors

    @staticmethod
    def run_multiple_times(data_1, data_2, num_runs=100, k=8):
        """
        Run AdaBoost multiple times and return averaged empirical and true errors.
        """
        all_empirical_errors = []
        all_true_errors = []

        for _ in range(num_runs):
            model = Adaboost(k)
            model.fit(data_1, data_2)
            empirical_errors, true_errors = model.compute_errors()
            all_empirical_errors.append(empirical_errors)
            all_true_errors.append(true_errors)

        avg_empirical_errors = np.mean(all_empirical_errors, axis=0)
        avg_true_errors = np.mean(all_true_errors, axis=0)

        return avg_empirical_errors, avg_true_errors
