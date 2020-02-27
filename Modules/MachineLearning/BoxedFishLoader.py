from torch.utils import data
from PIL import Image
import pandas as pd

class BoxedFishLoader(data.Dataset):
	def __init__(self, image_directory, label_csv_file):
		self.image_directory = image_directory
		self.label_csv_file = label_csv_file

		self.dt = pd.read_csv(self.label_csv_file)


	def __getitem__(self, index):
		