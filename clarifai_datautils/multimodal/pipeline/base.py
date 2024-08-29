import os
from typing import List, Type

from tqdm import tqdm

from .loaders import MultiModalLoader


class BaseTransform:
  """Base Transform Component"""

  def __init__(self) -> None:
    pass

  def __call__(self,) -> List:
    """Transform components."""
    raise NotImplementedError()


class Pipeline:
  """Text processing pipeline object from files"""

  def __init__(
      self,
      name: str,
      transformations: List[Type[BaseTransform]],
  ):
    """Initializes an Pipeline object.

    Args:
      name (str): Name of the pipeline.
      transformations (List): List of transformations to apply.

    """
    self.name = name
    self.transformations = transformations
    for transform in self.transformations:
      if not isinstance(transform, BaseTransform):
        raise ValueError('All transformations should be of type BaseTransform.')

    #TODO: Schema for transformations

  def run(self,
          files: str = None,
          folder: str = None,
          show_progress: bool = True,
          loader: bool = True):
    """Runs the Data Ingestion pipeline.

    Args:
        files Any[str,List]: List of files to process.
        folder (str): Folder containing the files.
        show_progress (bool): Whether to show progress bar.
        loader (bool): Whether to return a Clarifai Dataloader Object to pass to SDK Dataset Upload Functionality.

    Returns:
        List of transformed elements or ClarifaiDataLoader object.

    Example:
        >>> from clarifai_datautils.text import Pipeline
        >>> dataloader = Pipeline().run(files = 'xx.pdf', loader = True))
    """
    if files is None and folder is None:
      raise ValueError('Either files or folder should be provided.')
    if files and folder:
      raise ValueError('Provide any one of files or folder.')

    # Get files
    if files is not None:
      self.elements = [files] if isinstance(files, str) else files
      assert isinstance(self.elements, list), ' Files should be a list of strings.'
    elif folder is not None:
      self.elements = [os.path.join(folder, f) for f in os.listdir(folder)]

    # Apply transformations
    #TODO: num_workers support
    if show_progress:
      with tqdm(total=len(self.transformations), desc='Applying Transformations') as progress:
        self.elements = self.transformations[0](self.elements)
        if isinstance(self.elements, tuple):
          self.text_elements, self.image_elements = self.elements
        progress.update()

        if len(self.transformations) > 1:
          for transform in self.transformations[1:]:
            self.text_elements = transform(self.text_elements)
            progress.update()

    else:
      self.elements = self.transformations[0](self.elements)
      if isinstance(self.elements, tuple):
        self.text_elements, self.image_elements = self.elements

      if len(self.transformations) > 1:
        for transform in self.transformations[1:]:
          self.text_elements = transform(self.text_elements)

    self.elements = (self.text_elements, self.image_elements)

    if loader is True:
      return MultiModalLoader(elements=self.elements, pipeline_name=self.name)

    return self.elements

  def load() -> 'Pipeline':
    """Loads a pipeline from a yaml file.

    Returns:
        Pipeline object.

    """
    #TODO: Implement this
    pass

  def save(self,) -> None:
    """Saves the pipeline to a yaml file."""
    #TODO: Implement this
    pass

  def __str__(self) -> str:
    return (f"Pipeline: {self.name}\n"
            f"\tsize={len(self.transformations)}\n"
            f"\ttransformations={self.transformations}\n")
