import numpy as np
import matplotlib.pyplot as plt


def local_camp(X, w):
    return np.dot(X, w)


def activation_function(v):
    return 1 / (1 + np.exp(-v))


def derivative_activation_function(v):
    return activation_function(v) * (1 - activation_function(v))


def cost_function(X, y, w):
    v = local_camp(X, w)
    y_pred = activation_function(v)
    return 1/2 * np.mean((y - y_pred)**2)


def predict(X, w):
    return activation_function(local_camp(X, w)) > 0.5


def gradient_descent(X, y, w, learning_rate, epochs):
    for _ in range(epochs):
        v = local_camp(X, w)
        y_pred = activation_function(v)
        grad = - np.dot((y - y_pred) * X.T, derivative_activation_function(v))
        w = w - learning_rate * grad

    return w


def plot_plane_and_points(X, y, w):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generar puntos para graficar el plano
    x_vals = np.linspace(-0.5, 1.5, 50)
    y_vals = np.linspace(-0.5, 1.5, 50)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
    z_vals = w[0] * x_vals + w[1] * y_vals

    # Graficar el plano
    ax.plot_surface(x_vals, y_vals, z_vals, alpha=0.5)

    # Separar los puntos por clase
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    # Graficar puntos de cada clase con colores diferentes
    ax.scatter(class_0[:, 0], class_0[:, 1], y[y == 0], color='blue', label='Clase 0')
    ax.scatter(class_1[:, 0], class_1[:, 1], y[y == 1], color='red', label='Clase 1')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.legend()

    plt.title('Plano generado vs Puntos de entrada por clase')
    plt.show()


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    w = np.array([0.5, 0.5])
    epochs = 100
    learning_rate = 0.01
    w = gradient_descent(X, y, w, learning_rate, epochs)
    print("Parámetros finales del plano:", w)

    y_pred = predict(X, w)
    plot_plane_and_points(X, y, w)
