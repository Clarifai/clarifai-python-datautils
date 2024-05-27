from typing import TypeVar, Union

from clarifai_datautils.constants.base import DATASET_UPLOAD_TASKS

from .features import (TextFeatures, VisualClassificationFeatures, VisualDetectionFeatures,
                       VisualSegmentationFeatures)

OutputFeaturesType = TypeVar(
    'OutputFeaturesType',
    bound=Union[VisualClassificationFeatures, VisualDetectionFeatures, VisualSegmentationFeatures,
                TextFeatures])


class ClarifaiDataLoader:
  """Clarifai data loader base class."""

  def __init__(self) -> None:
    pass

  @property
  def task(self):
    raise NotImplementedError("Task should be one of {}".format(DATASET_UPLOAD_TASKS))

  def load_data(self) -> None:
    raise NotImplementedError()

  def __len__(self) -> int:
    raise NotImplementedError()

  def __getitem__(self, index: int) -> OutputFeaturesType:
    raise NotImplementedError()
