import numpy as np
import matplotlib.pyplot as plt
from utils import write_from_file, plot_perceptron, plot_adaboost
from Algorithms.Perceptron import Perceptron
from Algorithms.Adaboost import Adaboost

def main(data_txt):
    """
    Part I - Perceptron Algorithm

    This part of the code runs the Perceptron algorithm on different sets of flower data.

    Using the PerceptronClassifier:
    - fit function:
        * Trains the perceptron on the given data
    - get_weights function:
        * Returns the final weights vector
    - get_mistake_count function:
        * Returns the number of mistakes

    We perform classification on the following pairs of species:
    a. Setosa and Versicolor
    b. Setosa and Virginica
    c. Versicolor and Virginica
    """
    # Read data from file
    setosa, versicolor, virginica = write_from_file(data_txt)

    # Perceptron on Setosa and Versicolor
    print("a - Perceptron on Setosa and Versicolor:")
    perceptron_model = Perceptron()
    perceptron_model.fit(setosa, versicolor)

    w = perceptron_model.get_weights()
    num_of_mistakes = perceptron_model.get_mistake_count()

    # Plot results
    plot_perceptron(setosa, 'setosa', versicolor, 'versicolor', w)
    print(f'Final weights vector: {w}')
    print(f'Number of mistakes: {num_of_mistakes}')

    # Perceptron on Setosa and Virginica
    print("b - Perceptron on Setosa and Virginica:")
    perceptron_model.fit(setosa, virginica)

    w = perceptron_model.get_weights()
    num_of_mistakes = perceptron_model.get_mistake_count()

    # Plot results
    plot_perceptron(setosa, 'setosa', virginica, 'virginica', w)
    print(f'Final weights vector: {w}')
    print(f'Number of mistakes: {num_of_mistakes}')

    # Perceptron on Versicolor and Virginica
    print("c - Perceptron on Versicolor and Virginica:")
    print("Run time error - Cannot be classified linearly")
    # No need to train since perceptron cannot classify these classes linearly

    """
    Part II - Adaboost Algorithm

    This part of the code runs the AdaBoost algorithm on different sets of flower data.

    Using the AdaBoostClassifier:
    - fit function:
        * Trains the Adaboost model on the given data
    - run_multiple_times function:
        * Runs the Adaboost training multiple times and averages the errors

    We perform classification on the following pairs of species:
    a. Versicolor and Virginica
    """
    print("\nRun AdaBoost on Versicolor and Virginica")
    avg_empirical_errors, avg_true_errors = Adaboost.run_multiple_times(versicolor, virginica, 100)

    print("Empirical errors:", avg_empirical_errors)
    print("True errors:", avg_true_errors)

    # Plot results
    plot_adaboost(avg_empirical_errors, avg_true_errors)

if __name__ == "__main__":
    DATA_PATH = 'Dataset/iris.txt'  # Update with the correct path to your dataset
    main(DATA_PATH)
