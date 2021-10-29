# %%
import string
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List
from dataclasses import dataclass
from functools import lru_cache
from numpy.core.fromnumeric import argmax

# %%
@dataclass
class Line:
	r: float
	b: float

	def __call__(self, x: float) -> float:
		return np.math.tan(self.r) * x + self.b

	# L2 norm.
	# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula
	def distance_of_point_from_the_line(self, x: float, y: float) -> float:
		y_difference = np.abs(self(x) - y)
		slope_squared = np.math.pow(np.math.tan(self.r), 2)
		return y_difference / np.math.sqrt(1. + slope_squared)

	def to_numpy(self):
		return np.array([self.r, self.b])
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

	def __init__(self, wavelength: float, width: float, line: Line):
		self.wavelength = float(wavelength)
		self.line = line
		self.width = float(width)
		self.rgb = np.array(wavelength_to_RGB(self.wavelength))

	def __call__(self, i: int, j: int) -> np.ndarray:
		x, y = float(j), float(i)
		distance = self.line.distance_of_point_from_the_line(x, y)

		if distance <= self.width / 2.:
			return self.rgb
		if self.width/2. <= distance <= 5*self.width:
			return (
				np.math.sqrt(self.width) /
				np.math.pow(distance, 2) *
				self.rgb
			)
		
		return np.array([0., 0., 0.])

	def __repr__(self) -> str:
		return f"LaserBeam(wavelength={self.wavelength}, Line={str(self.line)}, width={self.width})"

	@staticmethod
	def from_numpy(theta: np.ndarray):
		return LaserBeam(
			wavelength=theta[0],
			line=Line(theta[1], theta[2]),
			width=theta[3],
		)

	def to_numpy(self):
		line = self.line
		return np.array([
			self.wavelength,
			line.r,
			line.b,
			self.width
		])

	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			return LaserBeam.from_numpy(other * self.to_numpy())
		if isinstance(other, np.ndarray):
			return LaserBeam.from_numpy(self.to_numpy() * other)
		if isinstance(other, list):
			return LaserBeam.from_numpy(self.to_numpy() * np.array(other))

	def __rmul__(self, other):
		return self * other

	def clip(self, min_params, max_params):
		clipped_params = np.clip(
			self.to_numpy(),
			min_params.to_numpy(),
			max_params.to_numpy()
		)
		self.wavelength = clipped_params[0]
		self.line = Line(clipped_params[1], clipped_params[2])
		self.width = clipped_params[3]

def add_images(image1, image2):
	if image1.shape != image2.shape:
		raise Exception("Wrong size")
	return np.clip(image1 + image2, 0, 1)


# %%

# %%
def add_laser_to_image2(image, laser):
	image = image.copy()
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			laser_image = laser(j,i)
			pixel = image[i,j].astype("float")/255. + laser(j,i)
			pixel = np.clip(pixel, 0.0, 1.0)
			pixel = pixel * 255
			image[i,j] = pixel.astype("uint8")
	return image
# %%
def generate_laser_image(laser_beam: LaserBeam, shape):
	laser_image = np.zeros(shape)
	for i in range(shape[0]):
		for j in range(shape[0]):
			rgb =  laser_beam(i,j)
			laser_image[i,j,0] = np.clip(rgb[0], 0, 1)
			laser_image[i,j,1] = np.clip(rgb[1], 0, 1)
			laser_image[i,j,2] = np.clip(rgb[2], 0, 1)
		
	return laser_image

# %%

def show_NRGB_image(image: np.ndarray):
	plt.imshow(image)

def save_NRGB_image(image: np.ndarray, number=0, name_length=5):
	ALPHABET = np.array(list(string.ascii_letters))
	im_name =f"attack/{number}_{''.join(np.random.choice(ALPHABET, size=name_length))}.jpg"
	plt.imsave(im_name, image)
# %%
