import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

class PeopleDataset:
	def __init__(self, size_range = 20):
		self.size_range = size_range
		self.X = None
		self.y = None
		self.generate_data()

	def generate_data(self):
		self.X = np.random.normal(175, 10, self.size_range).reshape(-1, 1)
		self.y = 0.45*self.X + 5 + np.random.normal(0, 5, self.size_range).reshape(-1, 1)

	def add_person(self, height, weight):
		self.X = np.append(self.X, [[height]], axis=0)
		self.y = np.append(self.y, [[weight]], axis=0)

	def remove_person(self, index):
		self.X = np.delete(self.X, index, axis=0)
		self.y = np.delete(self.y, index, axis=0)

	def get_coefficients(self):
		model = LinearRegression()
		model.fit(self.X, self.y)
		k = model.coef_[0][0]
		b = model.intercept_[0]
		return k, b

	def predict_height(self, weight):
		k, b = self.get_coefficients()
		height = (weight - b) / k
		return height

	def predict_weight(self, height):
		k, b = self.get_coefficients()
		weight = k * height + b
		return weight

	def print_regression_line(self):
		k, b = self.get_coefficients()
		plt.scatter(self.X, self.y)
		plt.plot(self.X, k*self.X + b, color='red')
		plt.xlabel('Height (cm)')
		plt.ylabel('Weight (kg)')
		plt.title('People Dataset with Regression Line')
		plt.show()

	def print_data(self):
		plt.scatter(self.X, self.y)
		plt.xlabel('Height (cm)')
		plt.ylabel('Weight (kg)')
		plt.title('People Dataset')
		plt.show()
