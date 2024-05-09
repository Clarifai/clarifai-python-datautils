from typing import List

from unstructured.partition.pdf import partition_pdf

from clarifai_datautils.constants.pipeline import MAX_CHARACTERS

from .base import BaseTransform


class PDFPartition(BaseTransform):
  """Partitions PDF file into text elements."""

  def __init__(self,
               ocr: bool = False,
               chunking_strategy: str = "basic",
               max_characters=MAX_CHARACTERS,
               overlap=None,
               overlap_all=True,
               **kwargs):
    """Initializes an PDFPartition object.

        Args:
            ocr (bool): Whether to use OCR.
            chunking_strategy (str): Chunking strategy to use.
            max_characters (int): Maximum number of characters in a chunk.
            overlap (int): Number of characters to overlap between chunks.
            overlap_all (bool): Whether to overlap all chunks.
            kwargs: Additional keyword arguments.
        """
    if chunking_strategy not in ["basic", "by_title"]:
      raise ValueError("chunking_strategy should be either 'basic' or 'by_title'.")
    self.chunking_strategy = chunking_strategy
    self.strategy = "fast"  #"ocr" if ocr else "fast"   #TODO: Add OCR Strategy and hi_res strategy
    self.max_characters = max_characters
    self.overlap = overlap
    self.overlap_all = overlap_all
    self.kwargs = kwargs

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

        Args:
            elements (List[str]): List of text elements.

        Returns:
            List of transformed text elements.

        """
    file_elements = []
    for filename in elements:
      file_element = partition_pdf(
          filename=filename,
          strategy=self.strategy,
          chunking_strategy=self.chunking_strategy,
          max_characters=self.max_characters,
          overlap=self.overlap,
          overlap_all=self.overlap_all,
          **self.kwargs)
      file_elements.extend(file_element)
      del file_element

    return file_elements
