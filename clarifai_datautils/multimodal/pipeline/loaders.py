import base64

from clarifai_datautils.constants.base import DATASET_UPLOAD_TASKS

from ...base import ClarifaiDataLoader
from ...base.features import TextFeatures, VisualClassificationFeatures


class MultiModalLoader():
  """MultiModal Dataset object."""

  def __init__(self, elements, pipeline_name=None):
    """
        Args:
          elements: Tuple of List of elements, where element[0]=text chunks,
          element[1]=image objects.
        """
    self.elements = elements
    self.pipeline_name = pipeline_name

  class TextDataLoader(ClarifaiDataLoader):
    """Text Dataset object."""

    def __init__(self, elements, pipeline_name=None):
      """
            Args:
              elements: List of elements.
            """
      self.elements = elements  #List of text chunk objects
      self.pipeline_name = pipeline_name

    @property
    def task(self):
      return DATASET_UPLOAD_TASKS.TEXT_CLASSIFICATION

    def __getitem__(self, index: int):
      meta = self.elements[index].metadata.to_dict()
      meta.pop('coordinates', None)
      meta.pop('detection_class_prob', None)
      if 'type' in self.elements[index].to_dict():
        if self.elements[index].to_dict()['type'] == 'Table':
          meta['type'] = 'Table'
      return TextFeatures(
          text=self.elements[index].text, labels=[self.pipeline_name], metadata=meta)

    def __len__(self):
      return len(self.elements)

  class VisualDataLoader(ClarifaiDataLoader):
    """Visual Dataset object."""

    def __init__(self, elements, pipeline_name=None):
      """
            Args:
              elements: List of elements.
            """
      self.elements = elements  #List of image objects
      self.pipeline_name = pipeline_name

    @property
    def task(self):
      return DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION

    def __getitem__(self, index: int):
      meta = self.elements[index].metadata.to_dict()
      meta.pop('detection_class_prob', None)
      meta.pop('coordinates', None)
      return VisualClassificationFeatures(
          image_path=None,
          image_bytes=base64.b64decode(meta.pop('image_base64', None)),
          labels=[self.pipeline_name],
          metadata=meta)

    def __len__(self):
      return len(self.elements)

  def get_loader(self, loader_type):
    if loader_type == 'text':
      return self.TextDataLoader(self.elements[0], self.pipeline_name)
    elif loader_type == 'image':
      return self.VisualDataLoader(self.elements[1], self.pipeline_name)


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
