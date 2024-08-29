from typing import List

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from clarifai_datautils.constants.pipeline import MAX_CHARACTERS

from .base import BaseTransform


class PDFExtraction(BaseTransform):
  """Extracts multimodal(text and image) from PDF file."""

  def __init__(
      self,
      extract_images_in_pdf=True,
      extract_image_block_types=["Image"],  # Can include Table in future if needed
      extract_image_block_to_payload=True,
      chunking_strategy: str = "basic",
      max_characters=MAX_CHARACTERS,
      overlap=None,
      overlap_all=True,
      **kwargs):
    """Initializes an PDFExtraction object.

     Args:
         extract_images_in_pdf (bool): Whether to extract images in PDF.
         extract_image_block_types (List): List of image block types to extract.
         extract_image_block_to_payload (bool): If 'True' returns image as bytes.
         chunking_strategy (str): Chunking strategy to use.
         max_characters (int): Maximum number of characters in a chunk.
         overlap (int): Number of characters to overlap between chunks.
         overlap_all (bool): Whether to overlap all chunks.
         kwargs: Additional keyword arguments.

     """

    if chunking_strategy not in ["basic", "by_title"]:
      raise ValueError("chunking_strategy should be either 'basic' or 'by_title'.")

    self.extract_images_in_pdf = extract_images_in_pdf
    self.extract_image_block_types = extract_image_block_types
    self.extract_image_block_to_payload = extract_image_block_to_payload
    self.chunking_strategy = chunking_strategy
    self.strategy = "hi_res"  #Always hi_res for PDF, if images/tables needs to be extracted.
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
    text_elements = []
    image_elements = []
    for filename in elements:
      file_element = partition_pdf(
          filename=filename,
          strategy=self.strategy,
          extract_images_in_pdf=self.extract_images_in_pdf,
          extract_image_block_types=self.extract_image_block_types,
          extract_image_block_to_payload=self.extract_image_block_to_payload,
          **self.kwargs)

      text_chunks = chunk_by_title(
          [elem for elem in file_element if elem.to_dict()['type'] != 'Image'],
          max_characters=self.max_characters,
          overlap=self.overlap,
          overlap_all=self.overlap_all,
          **self.kwargs)
      multimodal_elements = [elem for elem in file_element if elem.to_dict()['type'] == 'Image']
      text_elements.extend(text_chunks)
      image_elements.extend(multimodal_elements)

    return (text_elements, image_elements)
