import numpy as np
import matplotlib.pyplot as plt


def write_from_file(data_txt):
    """
    Reads the dataset from a text file and extracts three species: Setosa, Versicolor, and Virginica.
    Returns them as numpy arrays.
    """
    setosa, versicolor, virginica = [], [], []
    
    with open(data_txt, 'r') as file:
        for line in file:
            columns = line.split()
            sample = [float(columns[1]), float(columns[2])]
            
            if columns[4] == 'Iris-setosa':
                setosa.append(sample)
            elif columns[4] == 'Iris-versicolor':
                versicolor.append(sample)
            elif columns[4] == 'Iris-virginica':
                virginica.append(sample)

    return np.array(setosa), np.array(versicolor), np.array(virginica)

def plot_perceptron(data_1, name_1, data_2, name_2, w):
    """
    Plots the data points for two species along with the perceptron's decision boundary.
    """
    plt.scatter(data_1[:, 0], data_1[:, 1], color='red', label=f'Class 1: {name_1}')
    plt.scatter(data_2[:, 0], data_2[:, 1], color='blue', label=f'Class 2: {name_2}')

    # Calculate the decision boundary: w1*x + w2*y = 0
    if w[1] != 0:  
        x_vals = np.linspace(min(data_1[:, 0].min(), data_2[:, 0].min()), max(data_1[:, 0].max(), data_2[:, 0].max()))
        y_vals = -(w[0] / w[1]) * x_vals  # Solving for y in terms of x using the hyperplane equation
        plt.plot(x_vals, y_vals, color='black', label='Decision Boundary (Perceptron)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Perceptron: Decision Boundary')
    plt.show()

def plot_adaboost(avg_empirical_errors, avg_true_errors):
    
    X_values = list(range(1, len(avg_empirical_errors) + 1))
    
    plt.plot(X_values, avg_empirical_errors, label="Empirical Error", color="blue")
    plt.plot(X_values, avg_true_errors, label="True Error", color="red")

    plt.title('Adaboost results')
    plt.xlabel('Number of Classifications')
    plt.ylabel('Error Rate')

    plt.legend()
    plt.show()