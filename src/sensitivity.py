import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def getsensitivity(X, y, all_graphs = False, last = False):
  X = np.array(X)
  y = np.array(y)
  # Находим истинные коэффициенты для функции линейной регрессии
  reg = LinearRegression().fit(X, y)
  true_k, true_b = reg.coef_[0][0], reg.intercept_[0]

  deltas_k = []
  deltas_b = []

  for i in range(len(X)):
    X_without_i = np.delete(X, i, axis=0)
    y_without_i = np.delete(y, i, axis=0)
    reg = LinearRegression().fit(X_without_i, y_without_i)
    delta_k, delta_b = abs(true_k - reg.coef_[0][0]), abs(true_b - reg.intercept_[0])
    deltas_k.append(delta_k)
    deltas_b.append(delta_b)

    # Рисуем график
    if all_graphs == True:
      x_plot = np.linspace(150, 190, 2)
      y_plot = reg.predict(x_plot.reshape(-1, 1))
      plt.scatter(X_without_i, y_without_i)
      plt.plot(x_plot, y_plot, label=f'Removed {i+1}, k={reg.coef_[0][0]:.2f}, b={reg.intercept_[0]:.2f}')
      plt.legend()
      plt.show()

  # Рисуем графики
  if last == True:
    plt.subplot(221)
    plt.plot(X, y, 'o')
    plt.plot(X, reg.predict(X), 'r')
    plt.title("Исходная выборка")
    plt.xlabel("Рост")
    plt.ylabel("Вес")

    plt.subplot(222)
    plt.bar(range(len(X)), deltas_k)
    plt.title("Отличие коэффициента k")
    plt.xlabel("Индекс удаленного человека")
    plt.ylabel("Отличие")

    plt.subplot(223)
    plt.bar(range(len(X)), deltas_b)
    plt.title("Отличие коэффициента b")
    plt.xlabel("Индекс удаленного человека")
    plt.ylabel("Отличие")

    plt.tight_layout()
    plt.show()

  # Находим максимальные разницы в коэффициентах при удалении одного человека
  max_delta_k, max_delta_b = max(deltas_k), max(deltas_b)

  # Выводим результаты
  print(f'Max delta k: {max_delta_k:.2f}')
  print(f'Max delta b: {max_delta_b:.2f}')

  print(f'True k: {true_k:.2f}')
  print(f'True b: {true_b:.2f}')

  return max_delta_k, max_delta_b