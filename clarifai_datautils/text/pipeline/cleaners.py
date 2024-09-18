from typing import List

from unstructured.cleaners.core import (
    bytes_string_to_string, clean_bullets, clean_dashes, clean_extra_whitespace,
    clean_non_ascii_chars, clean_ordered_bullets, clean_postfix, clean_prefix,
    group_broken_paragraphs, remove_punctuation, replace_unicode_quotes)

from .base import BaseTransform


class Clean_extra_whitespace(BaseTransform):
  """Cleans extra whitespace from text."""

  def __init__(self):
    """Initializes an clean_extra_whitespace object."""
    pass

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_extra_whitespace(element.text)
    return elements


class Replace_unicode_quotes(BaseTransform):
  """Replaces unicode quotes with ASCII quotes."""

  def __init__(self):
    """Initializes an replace_unicode_quotes object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = replace_unicode_quotes(element.text)
    return elements


class Clean_dashes(BaseTransform):
  """Cleans dashes from text."""

  def __init__(self):
    """Initializes an clean_dashes object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_dashes(element.text)
    return elements


class Clean_bullets(BaseTransform):
  """Cleans bullets from text."""

  def __init__(self):
    """Initializes an clean_bullets object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_bullets(element.text)
    return elements


class Group_broken_paragraphs(BaseTransform):
  """Groups broken paragraphs."""

  def __init__(self):
    """Initializes an group_broken_paragraphs object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = group_broken_paragraphs(element.text)
    return elements


class Remove_punctuation(BaseTransform):
  """Removes punctuation from text."""

  def __init__(self):
    """Initializes an remove_punctuation object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = remove_punctuation(element.text)
    return elements


class Bytes_string_to_string(BaseTransform):
  """Converts bytes string to string."""

  def __init__(self):
    """Initializes an bytes_string_to_string object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      try:
        element.text = bytes_string_to_string(element.text)
      except ValueError:
        continue
    return elements


class Clean_non_ascii_chars(BaseTransform):
  """Cleans non-ASCII characters from text."""

  def __init__(self):
    """Initializes an clean_non_ascii_chars object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_non_ascii_chars(element.text)
    return elements


class Clean_ordered_bullets(BaseTransform):
  """Cleans ordered bullets from text."""

  def __init__(self):
    """Initializes an clean_ordered_bullets object."""

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_ordered_bullets(element.text)
    return elements


class Clean_prefix(BaseTransform):

  def __init__(self, pattern: str, ignore_case: bool = False, strip: bool = True):
    """Initializes an clean_prefix object.

    Args:
        pattern (str): The pattern for the prefix. Can be a simple string or a regex pattern.
        ignore_case (bool): Whether to ignore case.
        strip (bool): If True, removes leading whitespace from the cleaned string.

    """
    self.pattern = pattern
    self.ignore_case = ignore_case
    self.strip = strip

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_prefix(
          element.text, pattern=self.pattern, ignore_case=self.ignore_case, strip=self.strip)
    return elements


class Clean_postfix(BaseTransform):

  def __init__(self, pattern: str, ignore_case: bool = False, strip: bool = True):
    """Removes the postfix from the text.

    Args:
        pattern (str): The pattern for the postfix. Can be a simple string or a regex pattern.
        ignore_case (bool): Whether to ignore case.
        strip (bool): If True, removes leading whitespace from the cleaned string.

    """
    self.pattern = pattern
    self.ignore_case = ignore_case
    self.strip = strip

  def __call__(self, elements: List[str]) -> List[str]:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    for element in elements:
      element.text = clean_postfix(
          element.text, pattern=self.pattern, ignore_case=self.ignore_case, strip=self.strip)
    return elements
