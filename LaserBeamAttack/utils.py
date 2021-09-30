# %%
from typing import Union, List
import numpy as np
from dataclasses import dataclass

from numpy.core.fromnumeric import argmax

# %%
@dataclass
class Line:
	r: float
	b: float

	def __call__(self, x: float) -> float:
		return np.math.tan(self.r) * x + self.b

# %%
@dataclass
class Range:
	left: float 
	right: float

	def __contains__(self, value):
		return self.left <= value < self.right

	@property
	def length(self):
		return self.right - self.left
# %%
def wavelength_to_RGB(wavelength: Union[float, int]) -> List[float]:
	wavelength = float(wavelength)
	range1 = Range(380, 440)
	range2 = Range(440, 490)
	range3 = Range(490, 510)
	range4 = Range(510, 580)
	range5 = Range(580, 645)
	range6 = Range(645, 780)

	if wavelength in range1:
		R = (range1.right - wavelength) / range1.length
		G = 0.
		B = 1.
		return [R, G, B] 
	if wavelength in range2:
		R = 0.
		G = (wavelength - range2.left) / range2.length
		B = 1.
		return [R, G, B] 
	if wavelength in range3:
		R = 0.
		G = 1.
		B = (range3.right - wavelength) / range3.length
		return [R, G, B] 
	if wavelength in range4:
		R = (wavelength - range4.left) / range4.length
		G = 1.
		B = 0.
		return [R, G, B] 
	if wavelength in range5:
		R = 1.
		G = (range5.right - wavelength) / range5.length
		B = 0.
		return [R, G, B] 
	if wavelength in range6:
		R = 1.
		G = 0.
		B = 0.
		return [R, G, B] 

	return [0., 0., 0.]
	
# %%
wavelength_to_RGB(590)

# %%

class LaserBeam:

	def __init__(self, wavelength: int, width: int, line: Line):
		self.wavelength = float(wavelength)
		self.width = float(width)
		self.line = line
		self.rgb = np.array(wavelength_to_RGB(self.wavelength))

	def __call__(self, x: float, y: float) -> np.ndarray:
		distance = self.distance_of_point_from_the_line(x, y)

		if distance <= self.width / 2.:
			return self.rgb
		if self.width/2. <= distance <= 5*self.width:
			return (
				np.math.sqrt(self.width) /
				np.math.pow(distance, 2) *
				self.rgb
			)
		
		return np.array([0., 0., 0.])
			
	# L2 norm.
	# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula
	def distance_of_point_from_the_line(self, x: float, y: float) -> float:
		y_difference = np.abs(self.line(x) - y)
		slope_squared = np.math.pow(np.math.tan(self.line.r), 2)
		return y_difference / np.math.sqrt(1. + slope_squared)

# %%
def add_laser_to_image(image, laser):
	image = image.copy()
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			pixel = image[i,j].astype("float")/255. + laser(j,i)
			pixel = pixel * 255
			for idx, value in enumerate(pixel):
				pixel[idx] = value if value < 255 else 255
			image[i,j] = pixel.astype("uint8")
	return image