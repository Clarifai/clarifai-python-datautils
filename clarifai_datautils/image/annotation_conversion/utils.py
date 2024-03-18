import json
import os
from typing import Tuple

import PIL
from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from clarifai_datautils.errors import AnnotationsDatasetError, AnnotationsFormatError


class Clarifai_to_Datumaro():

  def __init__(
      self,
      main_path: str,
  ):
    """Converts a clarifai dataset to a Datumaro dataset.

    Args:
        path (str): The path to the clarifai dataset.

    """
    self.main_path = main_path
    self.image_list = os.listdir(os.path.join(self.main_path, 'inputs'))
    self.annotations_list = os.listdir(os.path.join(self.main_path, 'annotations'))
    self.label_map = {}

  def convert(self) -> Dataset:
    """Check folder format and creates a Datumaro Dataset.

    Returns:
        A Datumaro dataset object.
    """
    self.check_folder()
    # create a dataset
    dataset = Dataset.from_iterable(
        iterable=[self.create_item(path) for path in self.image_list],
        media_type=Image,
        categories=list(self.label_map.keys()))

    return dataset

  def create_item(self, image_path: str) -> DatasetItem:
    """Creates a Datumaro item from an image path."""
    image_full_path = os.path.join(self.main_path, 'inputs', image_path)
    image_data = Image.from_file(image_full_path)
    width, height = PIL.Image.open(image_full_path).size
    try:
      with open(
          os.path.join(self.main_path, 'annotations', image_path.split('.png')[0] + '.json'),
          'r') as file:
        item_data = json.load(file)
      # create annotations
      annotations = []
      for annot in item_data:
        #check if the annotation has a bounding box
        if 'regionInfo' in annot.keys() and 'boundingBox' in annot['regionInfo'].keys():
          x, y, w, h = self.clarifai_bbox_to_datumaro_bbox(annot['regionInfo']['boundingBox'],
                                                           width, height)
          label = annot['data']['concepts'][0]['name']
          value = self.label_map.get(label, len(self.label_map))
          self.label_map[label] = value
          annotations.append(Bbox(x=x, y=y, w=w, h=h, label=value))

    except FileNotFoundError:
      annotations = []

    return DatasetItem(id=image_path.split('.png')[0], media=image_data, annotations=annotations)

  def clarifai_bbox_to_datumaro_bbox(self, clarifai_bbox, width, height) -> Tuple[int]:
    left_col = clarifai_bbox['leftCol'] * width
    top_row = clarifai_bbox['topRow'] * height
    right_col = clarifai_bbox['rightCol'] * width
    bottom_row = clarifai_bbox['bottomRow'] * height

    obj_box = (left_col, top_row, right_col - left_col, bottom_row - top_row)
    return obj_box

  def check_folder(self):
    """Checks the clarifai folder format."""
    if not os.path.exists(self.main_path):
      raise AnnotationsDatasetError(f'Folder not found at {self.main_path}')

    if not os.path.exists(os.path.join(self.main_path, 'inputs')):
      raise AnnotationsFormatError(
          f'Folder does not contain an "inputs" folder at {self.main_path}')
    if not os.path.exists(os.path.join(self.main_path, 'annotations')):
      raise AnnotationsFormatError(
          f'Folder does not contain an "annotations" folder at {self.main_path}')

    if not all(img.endswith('.png') for img in self.image_list):
      raise AnnotationsFormatError(f'Folder should only contain images at {self.main_path}/inputs')
    if not all(img.endswith('.json') for img in self.annotations_list):
      raise AnnotationsFormatError(
          f'Folder should only contain annotations at {self.main_path}/annotations')
