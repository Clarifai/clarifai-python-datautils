import os.path as osp

from clarifai_datautils.multimodal import MarkdownPartition, Pipeline
from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
from clarifai_datautils.multimodal.pipeline.extractors import ExtractTextAfter

MARKDOWN_FILE_PATH = osp.abspath(osp.join(osp.dirname(__file__), "assets", "markdown-sample.md"))


class TestMarkdownPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline
    """

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            MarkdownPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            MarkdownPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractTextAfter(key='text_after', string='will be converted to an ellipsis. ')
        ])
    elements = pipeline.run(files=MARKDOWN_FILE_PATH)
    assert len(elements) == 4
    assert elements[0].text[:9] == 'An h1 hea'
    assert elements[0].metadata['filename'] == 'markdown-sample.md'
    assert elements[0].metadata['text_after'] == 'Unicode is supported. ☺'

  def test_pipeline_run_chunker(self,):
    """Tests for pipeline run with chunker"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            MarkdownPartition(chunking_strategy="by_title", max_characters=100),
            Clean_extra_whitespace(),
        ])
    elements = pipeline.run(files=MARKDOWN_FILE_PATH)
    assert len(elements) == 43
    assert elements[0].metadata['filename'] == 'markdown-sample.md'
    assert elements[0].metadata['languages'] == ['eng']
