from clarifai_datautils.constants.pipeline import *  # noqa: F403
from clarifai_datautils.multimodal.pipeline.cleaners import (Clean_extra_whitespace,
                                                             Group_broken_paragraphs)
from clarifai_datautils.multimodal.pipeline.Docx import DocxPartition
from clarifai_datautils.multimodal.pipeline.extractors import (ExtractDateTimeTz,
                                                               ExtractEmailAddress)
from clarifai_datautils.multimodal.pipeline.Markdown import MarkdownPartition
from clarifai_datautils.multimodal.pipeline.PDF import PDFPartition
from clarifai_datautils.multimodal.pipeline.Text import TextPartition


class Custom_Pipelines:
  """Text processing pipeline object from files"""

  def basic_pdf_pipeline():
    return [
        PDFPartition(),
    ]

  def standard_pdf_pipeline():
    return [
        PDFPartition(),
        Clean_extra_whitespace(),
        Group_broken_paragraphs(),
    ]

  def context_overlap_pdf_pipeline():
    return [
        PDFPartition(max_characters=5024, overlap=524),
        Clean_extra_whitespace(),
    ]

  def ocr_pdf_pipeline():
    return [
        PDFPartition(ocr=True),
    ]

  def structured_pdf_pipeline():
    return [
        PDFPartition(max_characters=1024, overlap=None),
        Clean_extra_whitespace(),
        ExtractDateTimeTz(),
        ExtractEmailAddress(),
    ]

  def standard_text_pipeline():
    return [
        TextPartition(max_characters=1024, overlap=None),
        Clean_extra_whitespace(),
        Group_broken_paragraphs(),
    ]

  def standard_docx_pipeline():
    return [
        DocxPartition(max_characters=1024, overlap=None),
        Clean_extra_whitespace(),
        Group_broken_paragraphs(),
    ]

  def standard_markdown_pipeline():
    return [
        MarkdownPartition(max_characters=1024, overlap=None),
        Clean_extra_whitespace(),
        Group_broken_paragraphs(),
    ]
