import os.path as osp

from clarifai_datautils.multimodal import DocxPartition, Pipeline
from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
from clarifai_datautils.multimodal.pipeline.extractors import ExtractTextAfter

DOCX_FILE_PATH = osp.abspath(osp.join(osp.dirname(__file__), "assets", "DOCX_TestPage.docx"))


class TestDocxPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline
    """

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            DocxPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            DocxPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractTextAfter(key='text_after', string='Test Complete,')
        ])
    elements = pipeline.run(files=DOCX_FILE_PATH)
    assert len(elements) == 1
    assert elements[0].text[:9] == 'Test Page'
    assert elements[0].metadata['filename'] == 'DOCX_TestPage.docx'
    assert elements[0].metadata['text_after'] == 'you may close this File.'

  def test_pipeline_run_chunker(self,):
    """Tests for pipeline run with chunker"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            DocxPartition(chunking_strategy="by_title", max_characters=100),
            Clean_extra_whitespace(),
        ])
    elements = pipeline.run(files=DOCX_FILE_PATH)
    assert len(elements) == 6
    assert elements[0].metadata['filename'] == 'DOCX_TestPage.docx'
    assert elements[0].metadata['languages'] == ['eng']
