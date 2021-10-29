# %%
from abc import abstractmethod
from functools import cached_property
from tokenize import Number
from matplotlib.pyplot import step
import numpy as np
from numpy import random
from numpy.lib.arraysetops import isin
from utils import LaserBeam, add_images, wavelength_to_RGB, Line, generate_laser_image
from dataclasses import dataclass
from utils import Range
from typing import Optional, Tuple
import logging 
import matplotlib.pyplot as plt
import matplotlib
import string
from PIL import Image

# %%
class GreedySearchOptimizerMeta:

	@abstractmethod
	def get_laser(self) -> LaserBeam: 
		pass

	@abstractmethod
	def update_params(self, params: LaserBeam, *args, **kwargs) -> LaserBeam:
		pass

	@abstractmethod
	def update_image(
		self, 
		image: np.ndarray, 
		params: LaserBeam
	) -> np.ndarray:
		pass

# %%
class LaserBeamOptimizer(GreedySearchOptimizerMeta):

	def __init__(
		self, 
		min_params: LaserBeam, 
		max_params: LaserBeam, 
		step_size: int=20
	) -> None:
		self.min_params = min_params
		self.max_params = max_params
		self.step_size = step_size

	def get_params(self) -> LaserBeam:
		random_params = (
			self.min_params.to_numpy() 
			+ np.random.uniform(0, 1) 
			* (self.max_params.to_numpy() - self.min_params.to_numpy())
		)

		return LaserBeam.from_numpy(random_params)

	def update_params(self, params: LaserBeam, sign) -> LaserBeam:
		s = np.random.uniform(1, self.step_size)
		q = s * self._draw_step()
		theta_prim = LaserBeam.from_numpy(
			params.to_numpy() + sign*q
		)
		theta_prim.clip(self.min_params, self.max_params)
		return theta_prim

	def update_image(
		self, 
		image: np.ndarray, 
		laser_beam: LaserBeam
	) -> np.ndarray:
		laser_image = generate_laser_image(laser_beam, image.shape[1:])
		return self._add_images(image, np.expand_dims(laser_image, 0))

	def _draw_step(self) -> np.ndarray:
		# is it best selection? is no selection better?
		rad = 1./360. * 2*np.math.pi
		Q = np.asfarray([
			[1,0,0,0],
			[0,rad,0,0],
			[0,0,1,0],
			[0,0,0,1],
			[1,rad,0,0],
			[1,0,1,0],
			[1,0,0,1],
			[0,rad,1,0],
			[0,rad,0,1],
			[0,0,1,1]
		])

		return Q[np.random.choice(len(Q))]
	
	def _add_images(
		self, 
		image1: np.ndarray, 
		image2: np.ndarray
	) -> np.ndarray:
		return add_images(image1, image2)

# %%

class GreedySearchPerturbationAttackAdapter:

	def __init__(self, model, optimizer: GreedySearchOptimizerMeta) -> None:
		self.model = model
		self.optimizer = optimizer

	def attack(
		self, 
		image: np.ndarray, 
		actual_class_confidence: float, 
		actual_class: int,
		t_max: int
	) -> Tuple[Optional[LaserBeam], Optional[int]]:
		params = self.optimizer.get_params()
		for i in range(t_max):
			predicted_class = actual_class
			for sign in [-1, 1]:
				params_prim = self.optimizer.update_params(params, sign)
				adversarial_image= self.optimizer.update_image(image, params)
				prediction = self.model.predict(adversarial_image)
				# save_NRGB_image(np.squeeze(adversarial_image, 0), i)
				predicted_class = prediction.argmax()
				actual_class_confidence_adv = prediction[0][actual_class]
				
				if actual_class_confidence < actual_class_confidence_adv:
					params = params_prim
					actual_class_confidence = \
						actual_class_confidence_adv
					break

			if predicted_class != actual_class:
				return params, predicted_class
		
		return None, None


# %%
class LaserBeamAttackAadapter(GreedySearchPerturbationAttackAdapter):

	def __init__(self, model, optimizer) -> None:
		super().__init__(model, optimizer)

	def run(
		self, 
		image: np.ndarray, 
		iterations: int,
		random_initializations: int
	) -> Tuple[Optional[LaserBeam], Optional[int]]:

		image = np.expand_dims(image, 0)
		prediction = self.model.predict(image)
		actual_class = prediction.argmax()
		conf_prim = prediction[0][actual_class]

		params = None
		for _ in range(random_initializations):
			params, predicted_class = self.attack(image, conf_prim, actual_class, iterations)
			logging.info(f"Adversarial params: {params}")
			if params is not None:
				return params, predicted_class
		logging.warning("Couldn't find adversarial laser parameters")
		return None, None