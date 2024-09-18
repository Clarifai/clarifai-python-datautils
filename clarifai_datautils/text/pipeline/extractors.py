from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from unstructured.cleaners.extract import (extract_datetimetz, extract_email_address,
                                           extract_ip_address, extract_ip_address_name,
                                           extract_text_after, extract_text_before)
from unstructured.documents.elements import Element, ElementMetadata

from clarifai_datautils.constants.pipeline import MAX_NODES, SKIP_NODES

from .base import BaseTransform


class LlamaIndexWrapper(BaseTransform):
  """ Wrapper class for LlamaIndex Extractor object. """

  def __init__(self, llama_extractor, max_nodes=MAX_NODES, skip_nodes=SKIP_NODES):
    """Initializes an LlamaIndexWrapper object.

    Args:
        llama_extractor (LlamaIndexExtractor): LlamaIndex Extractor object.
        max_nodes (int): Maximum number of nodes to extract.
        skip_nodes (int): Every nth node to extract.

    """
    self.max_nodes = max_nodes
    self.skip_nodes = skip_nodes
    self.llama_extractor = llama_extractor
    assert (self.llama_extractor.llm.to_dict()['class_name'] == 'ClarifaiLLM'
           ), "Only Clarifai LLM Models are allowed for extraction."
    self.parser = SentenceSplitter()

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    node_counter = 0
    for index, element in enumerate(elements):
      if node_counter == self.max_nodes:
        break
      if (index + 1) % self.skip_nodes == 0:
        node_counter = +1
        node = self.parser.get_nodes_from_documents([Document(text=element.text)])
        node = [self.llama_extractor(node)]
        extracted_element = Element(metadata=ElementMetadata.from_dict(node[0][0].metadata))
        extracted_element.text = node[0][0].text
        extracted_element.metadata.update(ElementMetadata.from_dict(element.metadata.to_dict()))
        elements[index] = extracted_element
    return elements


class ExtractDateTimeTz(BaseTransform):
  """Extracts datetime with timezone from text."""

  def __init__(self):
    """Initializes an ExtractDateTimeTz object."""
    pass

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      metadata = {'date_time': extract_datetimetz(element.text)}
      element.metadata.update(ElementMetadata.from_dict(metadata))
    return elements


class ExtractEmailAddress(BaseTransform):
  """Extracts email address from text."""

  def __init__(self):
    """Initializes an ExtractEmailAddress object."""
    pass

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      metadata = {'email_address': extract_email_address(element.text)}
      element.metadata.update(ElementMetadata.from_dict(metadata))
    return elements


class ExtractIpAddress(BaseTransform):
  """Extracts IP address from text."""

  def __init__(self):
    """Initializes an ExtractIpAddress object."""
    pass

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      metadata = {'ip_address': extract_ip_address(element.text)}
      element.metadata.update(ElementMetadata.from_dict(metadata))
    return elements


class ExtractIpAddressName(BaseTransform):
  """Extracts IP address with name from text."""

  def __init__(self):
    """Initializes an ExtractIpAddressName object."""
    pass

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      metadata = {'ip_address_name': extract_ip_address_name(element.text)}
      element.metadata.update(ElementMetadata.from_dict(metadata))
    return elements


class ExtractTextAfter(BaseTransform):
  """Extracts text after a given string."""

  def __init__(self, key: str, string: str):
    """Initializes an ExtractTextAfter object.

    Args:
        key (str): Key to store the extracted text.
        string (str): String to extract text after.

    """
    self.key = key
    self.string = string

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      try:
        metadata = {self.key: extract_text_after(element.text, self.string)}
        element.metadata.update(ElementMetadata.from_dict(metadata))
      except Exception:
        pass
    return elements


class ExtractTextBefore(BaseTransform):
  """Extracts text before a given string."""

  def __init__(self, key: str, string: str):
    """Initializes an ExtractTextBefore object.

    Args:
        key (str): Key to store the extracted text.
        string (str): String to extract text before.

    """
    self.key = key
    self.string = string

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      try:
        metadata = {self.key: extract_text_before(element.text, self.string)}
        element.metadata.update(ElementMetadata.from_dict(metadata))
      except Exception:
        pass
    return elements
