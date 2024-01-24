from typing import Any, Dict

from datumaro.components.dataset import Dataset
from datumaro.components.errors import (DatasetError, DatasetImportError, DatasetNotFoundError,
                                        MultipleFormatsMatchError)

from clarifai_utils.constants.annotations import (IMAGE_ANNOTATION_FORMATS,
                                                  IMAGE_ANNOTATION_FORMATS_TO_TASKS,
                                                  IMAGE_FORMAT_MAP)
from clarifai_utils.errors import AnnotationsDatasetError, AnnotationsFormatError
from clarifai_utils.image.annotation_conversion.base import ClarifaiDataLoader
from clarifai_utils.image.annotation_conversion.loaders import (ClassificationDataLoader,
                                                                DetectionDataLoader,
                                                                SegmentationDataLoader)


class ImageAnnotations():
  """Annotaions is a class that provides image annotation utilities."""

  def __init__(
      self,
      dataset_object: Dataset,
      annotation_format: str,
      task: str,
  ):
    """Initializes an Annotaions object.

      Args:
        dataset_object (Dataset): Datumaro Dataset Object.
        annotation_format (str): The format of the dataset.
        task (str): The task of the dataset.

      """
    self._dataset = dataset_object
    self.annotation_format = annotation_format
    self.task = task

  @classmethod
  def import_from(cls, path: str, format: str) -> Dataset:
    """Imports a dataset from a given path and format.

        Args:
            path (str): The path to the dataset.
            format (str): The format of the dataset.

        Returns:
            A dataset object.

        Example:
            >>> from clarifai_utils import ImageAnnotations
            >>> format = ImageAnnotations.import_from(path=folder_path, format = 'coco_detection')
        """
    if format not in IMAGE_ANNOTATION_FORMATS:
      raise AnnotationsFormatError(
          'Invalid format. Format must be one of {}'.format(IMAGE_ANNOTATION_FORMATS))
    #task of the dataset
    task = IMAGE_ANNOTATION_FORMATS_TO_TASKS[format]

    try:
      format_name = IMAGE_FORMAT_MAP[format]
      dataset = Dataset.import_from(path, format_name)
    except (DatasetError, DatasetImportError, DatasetNotFoundError) as ex:
      raise AnnotationsDatasetError(ex)

    return ImageAnnotations(dataset, format, task)

  def get_info(self,) -> Dict[str, Any]:
    """Gets information about a dataset.

        Returns:
            A dictionary containing the information about the dataset.

        Example:
            >>> from clarifai_utils import ImageAnnotations
            >>> format = ImageAnnotations.import_from(path=folder_path, format = 'coco_detection')
            >>> info = format.get_info()
        """
    return {
        'size': len(self._dataset._data),
        'source_path': self._dataset._source_path,
        'annotated_items_count': self._dataset.get_annotated_items(),
        'annotations_count': self._dataset.get_annotations(),
        'sub_folders': list(self._dataset.get_subset_info()),
        'categories': list(self._dataset.get_categories_info())
    }

  def export_to(self, path: str, format: str) -> None:
    """Exports a dataset to a given path and format.

        Args:
            path (str): The path to the dataset.
            format (str): The format of the dataset.

        Example:
            >>> from clarifai_utils import ImageAnnotations
            >>> format = ImageAnnotations.import_from(path=folder_path, format = 'coco_detection')
            >>> format.export_to(path=output_folder_path, format = 'voc_detection')
        """
    if format not in IMAGE_ANNOTATION_FORMATS:
      raise AnnotationsFormatError('Invalid format')

    try:
      format_name = IMAGE_FORMAT_MAP[format]
      self._dataset.export(path, format_name)
    except Exception as ex:
      raise AnnotationsDatasetError(ex)

  @staticmethod
  def detect_format(path: str) -> str:
    """Gets the format of a dataset.

        Args:
            path (str): The path to the dataset.

        Returns:
            The format of the dataset.

        Example:
            >>> from clarifai_utils import ImageAnnotations
            >>> format = ImageAnnotations.detect_format(path=folder_path)
        """
    try:
      dataset_format = Dataset.detect(path)
    except MultipleFormatsMatchError as e:
      raise AnnotationsFormatError(e)
    if dataset_format and dataset_format in IMAGE_FORMAT_MAP.values():
      reversed_format_map = dict([(value, key) for key, value in IMAGE_FORMAT_MAP.items()])
      dataset_format = reversed_format_map[dataset_format]
    if dataset_format and dataset_format not in IMAGE_ANNOTATION_FORMATS:
      raise AnnotationsFormatError('Given folder does not contain a supported dataset format')
    return dataset_format

  @property
  def dataloader(self) -> ClarifaiDataLoader:
    """Returns a Clarifai Dataloader Object to pass to SDK Dataset Upload Functionality.

        Returns:
            A ClarifaiDataloader object.

        Example:
            >>> from clarifai_utils import ImageAnnotations
            >>> format = ImageAnnotations.import_from(path=folder_path, format = 'coco_detection')
            >>> clarifai_dataset_loader = format.dataloader
        """
    if self.task == 'visual_classification':
      return ClassificationDataLoader(self._dataset)
    elif self.task == 'visual_detection':
      return DetectionDataLoader(self._dataset)
    elif self.task == 'visual_segmentation':
      return SegmentationDataLoader(self._dataset)

  def __str__(self) -> str:
    separator = "\t"
    return (f"Dataset\n"
            f"\tsize={len(self._dataset._data)}\n"
            f"\tsource_path={self._dataset._source_path}\n"
            f"\tannotated_items_count={self._dataset.get_annotated_items()}\n"
            f"\tannotations_count={self._dataset.get_annotations()}\n"
            f"subsets\n"
            f"\t{separator.join(self._dataset.get_subset_info())}"
            f"infos\n"
            f"\t{separator.join(self._dataset.get_infos())}"
            f"categories\n"
            f"\t{separator.join(self._dataset.get_categories_info())}")
