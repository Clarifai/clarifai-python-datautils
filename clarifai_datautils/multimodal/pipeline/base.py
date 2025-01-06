import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Type
from schema import And, Schema

from tqdm import tqdm

from .basetransform import BaseTransform
from .loaders import MultiModalLoader, TextDataLoader


def get_schema() -> Schema:
  """Initialize the schema for Data Ingestion Pipeline transformations.

        This schema validates:

        - transformations must be a list
        - First item in the list must be one of the following: PDFPartition, TextPartition, PDFPartitionMultimodal, DocxPartition, MarkdownPartition
        - Each item in the list must be of BaseTransform instance

        Returns:
            Schema: The schema for transformations.
        """
  return Schema(And(list, lambda x: x[0].__class__.__name__ in ['PDFPartition', 'TextPartition', 'PDFPartitionMultimodal', 'DocxPartition', 'MarkdownPartition'], lambda x: all(isinstance(item, BaseTransform) for item in x)), error="Invalid transformations data.")


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
    self.transformation_schema = get_schema()
    self.transformation_schema.validate(self.transformations)

  def run(self,
          files: str = None,
          folder: str = None,
          show_progress: bool = True,
          loader: bool = True,
          num_workers: int = 4):
    """Runs the Data Ingestion pipeline.

    Args:
        files Any[str,List]: List of files to process.
        folder (str): Folder containing the files.
        show_progress (bool): Whether to show progress bar.
        loader (bool): Whether to return a Clarifai Dataloader Object to pass to SDK Dataset Upload Functionality.
        num_workers (int): Number of workers to use for parallel processing.

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
      all_files = [files] if isinstance(files, str) else files
      assert isinstance(all_files, list), 'Files should be a list of strings.'
    elif folder is not None:
      all_files = [os.path.join(folder, f) for f in os.listdir(folder)]

    self.elements = []

    # Apply transformations
    def transform_file(file_elements):
      for transform in self.transformations:
        file_elements = transform(file_elements)
      self.elements.extend(file_elements)

    # num_workers support
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
      if show_progress:
        with tqdm(total=len(all_files), desc='Transforming Files') as progress:
          futures = [executor.submit(transform_file, [file]) for file in all_files]
          for job in futures:
            job.result()
            progress.update()
      else:
        futures = [executor.submit(transform_file, [file]) for file in all_files]
        for job in futures:
          job.result()

    if loader:
      if self.transformations[0].__class__.__name__ == 'PDFPartitionMultimodal':
        return MultiModalLoader(elements=self.elements, pipeline_name=self.name)
      return TextDataLoader(elements=self.elements, pipeline_name=self.name)

    return self.elements

  @classmethod
  def load(self, name) -> 'Pipeline':
    """Loads a ready to use set of pipelines.

    Returns:
        Pipeline object.

    """
    try:
      from clarifai_datautils.multimodal.pipeline.custom_pipeline import Custom_Pipelines
    except Exception as e:
      raise ImportError('cannot Import Custom_Pipelines') from e

    self.name = name
    self.custom_pipelines_map = {
        'basic_pdf': Custom_Pipelines.basic_pdf_pipeline(),
        'standard_pdf': Custom_Pipelines.standard_pdf_pipeline(),
        'context_overlap_pdf': Custom_Pipelines.context_overlap_pdf_pipeline(),
        'ocr_pdf': Custom_Pipelines.ocr_pdf_pipeline(),
        'structured_pdf': Custom_Pipelines.structured_pdf_pipeline(),
        'standard_text': Custom_Pipelines.standard_text_pipeline(),
        'standard_docx': Custom_Pipelines.standard_docx_pipeline(),
        'standard_markdown': Custom_Pipelines.standard_markdown_pipeline(),
    }
    try:
      if self.name in self.custom_pipelines_map:
        return Pipeline(self.name, self.custom_pipelines_map[self.name])

    except Exception as e:
      raise ValueError(f'Pipeline {self.name} not found in custom_pipelines_map.') from e

  def info(self,) -> None:
    """Prints the pipeline information."""
    if self.name in self.custom_pipelines_map:
      print(f'Pipeline: {self.name}')
      for transform in self.custom_pipelines_map[self.name]:
        print(f'\t{transform}')
    else:
      print(f'Pipeline: {self.name}')
      for transform in self.transformations:
        print(f'\t{transform}')

  def save(self,) -> None:
    """Saves the pipeline to a yaml file."""
    #TODO: Implement this
    pass

  def __str__(self) -> str:
    return (f"Pipeline: {self.name}\n"
            f"\tsize={len(self.transformations)}\n"
            f"\ttransformations={self.transformations}\n")
