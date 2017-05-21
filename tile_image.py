from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np


def toRGB(im):
	imRGB = np.repeat(im[:, :, np.newaxis], 3, axis=2)
	return imRGB


def read_image(image_path):
	return toRGB(misc.imread(image_path, flatten=False))


def draw_rectangle(draw, coordinates, color, width=1):
	for i in range(width):
		rect_start = (coordinates[0] - i, coordinates[1] - i)
		rect_end = (coordinates[2] + i, coordinates[3] + i)
		draw.rectangle((rect_start, rect_end), outline=color)


class SolarImageLabeler(object):

	def __init__(self, image_path, patch_size):
		self.image = read_image(image_path)
		print(self.image.shape)
		self.patch_size = patch_size
		self.labels = []

	def get_patch(self, i, j):
		x1 = i*self.patch_size
		x2 = (i*self.patch_size) + self.patch_size
		y1 = j*self.patch_size
		y2 = (j*self.patch_size) + self.patch_size
		return self.image[y1:y2, x1:x2]

	def add_label(self, i, j, type):
		color = "grey"
		if type == "AR":
			color = "red"
		elif type == "CH":
			color = "purple"
		elif type == "FL":
			color = "green"
		elif type == "SG":
			color = "blue"

		x1 = i*self.patch_size
		x2 = (i*self.patch_size) + self.patch_size
		y1 = j*self.patch_size
		y2 = (j*self.patch_size) + self.patch_size
		self.labels.append([[x1, y1, x2, y2], color])

	def save_fig(self, image_path):
		pil_image = Image.fromarray(self.image)
		draw = ImageDraw.Draw(pil_image)
		for l in self.labels:
			draw_rectangle(draw, l[0], l[1], width=5)
		pil_image.save(image_path)


def test_labeler():
	path = "/Users/ahmetkucuk/Documents/Developer/python/Scripts4Vision/resources/solarnet/images/2014_11_22__04_00_08_62__SDO_AIA_AIA_131.png"
	new_image = "/Users/ahmetkucuk/Documents/Developer/python/Scripts4Vision/resources/solarnet/images/2014_11_22__04_00_08_62__SDO_AIA_AIA_131_new.png"

	l = SolarImageLabeler(path, 256)

	patch = l.get_patch(8, 10)
	pil_image = Image.fromarray(patch)
	pil_image.save("test2.png")

	l.add_label(10, 10, "AR")
	l.add_label(9, 10, "CH")
	l.add_label(8, 10, "FL")
	l.add_label(7, 10, "SG")
	l.save_fig(new_image)
test_labeler()
