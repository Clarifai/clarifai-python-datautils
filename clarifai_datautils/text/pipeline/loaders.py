from ...base import ClarifaiDataLoader
from ...base.features import TextFeatures


class TextDataLoader(ClarifaiDataLoader):
  """Text Dataset object."""

  def __init__(self, elements, pipeline_name = None):
    """
    Args:
      elements: List of elements.
    """
    self.elements = elements
    self.pipeline_name = pipeline_name

  @property
  def task(self):
    return "text_classification" #TODO: Better dataset name in SDK

  def __getitem__(self, index: int):
    return TextFeatures(
        text=self.elements[index].text,
        labels = self.pipeline_name,
        metadata = self.elements[index].metadata.to_dict())

  def __len__(self):
    return len(self.elements)