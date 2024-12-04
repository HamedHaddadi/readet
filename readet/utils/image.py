# ########################################## #
# utilities to extract images from PDF files #
# and to digitize the image 			     #
# ########################################## #
from typing import List 
import matplotlib.pyplot as plt 
from mpl_point_clicker import clicker 


class Digitizer:
	
	def __init__(self, figure: plt.Axes, x_range: List[float],
					y_range: List[float], labels: List[str]) -> None:
		
		all_labels = ['corners']
		all_labels.extend(labels)
		self.clicker = clicker(figure, all_labels)
		self._x_range = abs(x_range[0] - x_range[1])
		self._y_range = abs(y_range[0] - y_range[1])
		self.labels = labels 
	
	@property 
	def x_range(self):
		return self._x_range 
	
	@x_range.setter 
	def x_range(self, new_range):
		if isinstance(new_range, (tuple, list)):
			self._x_range = abs(new_range[0] - new_range[1])
		elif isinstance(new_range, float):
			self._x_range = new_range 
	
	@property
	def y_range(self):
		return self._y_range 
	
	@y_range.setter 
	def y_range(self, new_range):
		if isinstance(new_range, (tuple, list)):
			self._y_range = abs(new_range[0] - new_range[1])
		elif isinstance(new_range, float):
			self._y_range = new_range 
	
	@property 
	def center(self):
		"""
		generates (0,0) points of the graph
		"""
		self.clicker._positions['corner'][1]
	
	@property 
	def delta_x(self):
		return self.clicker._positions['corner'][1][1] - self.clicker._positions['corner'][0][1]

	@property 
	def delta_y(self):
		return self.clicker._positions['corner'][2][0] - self.clicker._positions['corner'][1][0]

	@property 
	def dx(self):
		return self.x_range/self.delta_x 

	@property 
	def dy(self):
		return self.y_range/self.delta_y 	

	def tare(self):
		center = self.center 
		for label in self.labels:
			adj_values = [(point[0] - center[0], point[1] - center[1]) 
						for point in self.clicker._positions[label]]
			self.clicker._positions[label] = adj_values 
	
	def to_real(self):
		self.tare()
		dx = self.dx 
		dy = self.dy
		for label in self.labels:
			adj_values = [(point[0]*dx, point[1]*dy) for point in self.clicker._positions[label]]
			self.clicker._positions[label] = adj_values 
	 	
	def to_csv(self):
		pass 
	

	

		
			

			


	

		
		



