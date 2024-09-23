import base64

from clarifai_datautils.constants.base import DATASET_UPLOAD_TASKS

from ...base import ClarifaiDataLoader
from ...base.features import MultiModalFeatures, TextFeatures


class MultiModalLoader(ClarifaiDataLoader):
  """MultiModal Dataset object."""

  def __init__(self, elements, pipeline_name=None):
    """
        Args:
          elements: Tuple of List of elements, where element[0]=text chunks,
          element[1]=image objects.
        """
    self.elements = []
    self.pipeline_name = pipeline_name
    if isinstance(elements, tuple):
      self.elements.extend(elements[0])
      self.elements.extend(elements[1])

  @property
  def task(self):
    return DATASET_UPLOAD_TASKS.MULTIMODAL_DATASET

  def __getitem__(self, index: int):
    meta = self.elements[index].metadata.to_dict()
    meta.pop('coordinates', None)
    meta.pop('detection_class_prob', None)
    image_data = meta.pop('image_base64', None)
    if image_data is not None:
      # Ensure image_data is already bytes before encoding
      image_data = base64.b64decode(image_data)
      text = None
      meta['type'] = 'image'
    else:
      text = self.elements[index].text

    if self.elements[index].to_dict()['type'] == 'Table':
      meta['type'] = 'table'

    return MultiModalFeatures(
        text=text, image_bytes=image_data, labels=[self.pipeline_name], metadata=meta)

  def __len__(self):
    return len(self.elements)


class TextDataLoader(ClarifaiDataLoader):
  """Text Dataset object."""

  def __init__(self, elements, pipeline_name=None):
    """
    Args:
      elements: List of elements.
    """
    self.elements = elements
    self.pipeline_name = pipeline_name

  @property
  def task(self):
    return DATASET_UPLOAD_TASKS.TEXT_CLASSIFICATION  #TODO: Better dataset name in SDK

  def __getitem__(self, index: int):
    return TextFeatures(
        text=self.elements[index].text,
        labels=self.pipeline_name,
        metadata=self.elements[index].metadata.to_dict())

  def __len__(self):
    return len(self.elements)
