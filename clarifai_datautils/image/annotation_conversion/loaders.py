import re

import cv2
import datumaro.plugins.transforms as transforms
import numpy as np
from datumaro.components.annotation import AnnotationType
from datumaro.components.media import ImageFromNumpy
from clarifai_datautils.constants.base import DATASET_UPLOAD_TASKS

from ...base import ClarifaiDataLoader
from ...base.features import (VisualClassificationFeatures, VisualDetectionFeatures,
                              VisualSegmentationFeatures)

delimiters = [",", "|", ";", "/", "\\", ":", " "]


class ClassificationDataLoader(ClarifaiDataLoader):
  """Annotation's Classificatoin Dataset object."""

  def __init__(self, annotation_object):
    """
    Args:
      annotation_object: Annotation object.
    """
    self.annotation_object = annotation_object
    self.map_ids = {
        count: {
            'id': item.id,
            'subset': item.subset
        }
        for count, item in enumerate(annotation_object)
    }
    label_map = annotation_object.categories()[AnnotationType.label]._indices
    self.label_map = dict([(value, key)
                           for key, value in label_map.items()])  #swapped key and value

  @property
  def task(self):
    return DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION

  def __getitem__(self, index: int):
    dataset_item = self.annotation_object.get(
        id=self.map_ids[index]['id'], subset=self.map_ids[index]['subset'])

    image_path = dataset_item.media.path if type(dataset_item.media) != ImageFromNumpy else None
    # Some datasets have images stored as bytes instead of paths
    if dataset_item.media.bytes is None:
      _, encoded_image = cv2.imencode('.png', dataset_item.media.data)
      image_bytes = encoded_image.tobytes()
    else:
      image_bytes = dataset_item.media.bytes

    concept_ids = []
    annotations = dataset_item.annotations

    for ann in annotations:
      if ann.type == AnnotationType.label:
        concept_name = self.label_map[ann.label]
        concept_id = concept_name.lower().replace(' ', '-')
        concept_ids.append(concept_id)

    return VisualClassificationFeatures(
        image_path,
        concept_ids,
        id=re.split(', |_|-|!|:|/', dataset_item.id)[-1],
        image_bytes=image_bytes)

  def __len__(self):
    return len(self.annotation_object)


class DetectionDataLoader(ClarifaiDataLoader):
  """Annotation's Detection Dataset object."""

  def __init__(self, annotation_object):
    """
    Args:
      annotation_object: Annotation object.
    """
    self.annotation_object = annotation_object
    self.map_ids = {
        count: {
            'id': item.id,
            'subset': item.subset
        }
        for count, item in enumerate(annotation_object)
    }
    label_map = annotation_object.categories()[AnnotationType.label]._indices
    self.label_map = dict([(value, key)
                           for key, value in label_map.items()])  #swapped key and value

  @property
  def task(self):
    return DATASET_UPLOAD_TASKS.VISUAL_DETECTION

  def __getitem__(self, index: int):
    dataset_item = self.annotation_object.get(
        id=self.map_ids[index]['id'], subset=self.map_ids[index]['subset'])

    image_path = dataset_item.media.path
    image_bytes = dataset_item.media.bytes

    height, width = dataset_item.media.size  #height and width of image

    annots = []  # bboxes
    concept_ids = []
    annotations = dataset_item.annotations

    for ann in annotations:
      if ann.type == AnnotationType.bbox:

        # get concept info
        # note1: concept_name can be human readable
        # note2: concept_id can only be alphanumeric, up to 32 characters, with no special chars except `-` and `_`
        concept_name = self.label_map[ann.label]
        concept_id = concept_name.lower().replace(' ', '-')

        # get bbox information
        # note1: datumaro bboxes are `[x_min, y_min, width, height]` in pixels
        # note2: clarifai bboxes are `[x_min, y_min, x_max, y_max]` normalized between 0-1.0

        obj_box = ann.get_bbox()
        clarifai_bbox = {
            'left_col': max(0, obj_box[0] / width),
            'top_row': max(0, obj_box[1] / height),
            'right_col': min(1, (obj_box[0] + obj_box[2]) / width),
            'bottom_row': min(1, (obj_box[1] + obj_box[3]) / height)
        }
        if (clarifai_bbox['left_col'] >= clarifai_bbox['right_col']) or (
            clarifai_bbox['top_row'] >= clarifai_bbox['bottom_row']):
          continue
        annots.append([
            clarifai_bbox['left_col'], clarifai_bbox['top_row'], clarifai_bbox['right_col'],
            clarifai_bbox['bottom_row']
        ])
        concept_ids.append(concept_id)

    assert len(concept_ids) == len(annots), f"Num concepts must match num bbox annotations\
        for a single image. Found {len(concept_ids)} concepts and {len(annots)} bboxes."

    return VisualDetectionFeatures(
        image_path,
        concept_ids,
        annots,
        id=re.split(', |_|-|!|:|/', dataset_item.id)[-1],
        image_bytes=image_bytes)

  def __len__(self):
    return len(self.annotation_object)


class SegmentationDataLoader(ClarifaiDataLoader):
  """Annotation's Segmentation Dataset object."""

  def __init__(self, annotation_object):
    """
    Args:
      annotation_object: Annotation object.
    """
    self.annotation_object = annotation_object
    self.map_ids = {
        count: {
            'id': item.id,
            'subset': item.subset
        }
        for count, item in enumerate(annotation_object)
    }
    label_map = annotation_object.categories()[AnnotationType.label]._indices
    self.label_map = dict([(value, key)
                           for key, value in label_map.items()])  #swapped key and value

  @property
  def task(self):
    return DATASET_UPLOAD_TASKS.VISUAL_SEGMENTATION

  def __getitem__(self, index: int):
    dataset_item = self.annotation_object.get(
        id=self.map_ids[index]['id'], subset=self.map_ids[index]['subset'])

    image_path = dataset_item.media.path
    image_bytes = dataset_item.media.bytes

    height, width = dataset_item.media.size  #height and width of image

    annots = []  # bboxes
    concept_ids = []
    annotations = dataset_item.annotations

    for ann in annotations:
      if ann.type == AnnotationType.polygon:

        # get concept info
        # note1: concept_name can be human readable
        # note2: concept_id can only be alphanumeric, up to 32 characters, with no special chars except `-` and `_`
        concept_name = self.label_map[ann.label]
        concept_id = concept_name.lower().replace(' ', '-')

        poly = ann.get_points()
        poly = np.array(poly)
        poly[:, 0], poly[:, 1] = poly[:, 0] / width, poly[:, 1] / height
        poly = np.clip(poly, 0, 1)
        annots.append(poly.tolist())
        concept_ids.append(concept_id)

      if ann.type == AnnotationType.mask:
        annotat = transforms.MasksToPolygons.convert_mask(ann)
        for sub_ann in annotat:
          # get concept info
          # note1: concept_name can be human readable
          # note2: concept_id can only be alphanumeric, up to 32 characters, with no special chars except `-` and `_`
          concept_name = self.label_map[ann.label]
          concept_id = concept_name.lower().replace(' ', '-')

          # get bbox information
          # note1: datumaro bboxes are `[x_min, y_min, width, height]` in pixels
          # note2: clarifai bboxes are `[x_min, y_min, x_max, y_max]` normalized between 0-1.0

          poly = sub_ann.get_points()
          poly = np.array(poly)
          poly[:, 0], poly[:, 1] = poly[:, 0] / width, poly[:, 1] / height
          poly = np.clip(poly, 0, 1)
          annots.append(poly.tolist())
          concept_ids.append(concept_id)

    return VisualSegmentationFeatures(
        image_path,
        concept_ids,
        annots,
        id=re.split(', |_|-|!|:|/', dataset_item.id)[-1],
        image_bytes=image_bytes)

  def __len__(self):
    return len(self.annotation_object)
