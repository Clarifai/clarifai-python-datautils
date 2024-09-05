import os.path as osp

from clarifai_datautils.multimodal import Pipeline, TextPartition
from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
from clarifai_datautils.multimodal.pipeline.extractors import ExtractTextAfter

TEXT_FILE_PATH = osp.abspath(
    osp.join(osp.dirname(__file__), "assets", "book-war-and-peace-1p.txt"))


class TestPDFPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline
    """

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            TextPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            TextPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractTextAfter(
                key='text_after', string='grippe being then a new word in St. Petersburg,')
        ])
    elements = pipeline.run(files=TEXT_FILE_PATH)
    assert len(elements) == 4
    assert elements[0].text[:9] == 'CHAPTER I'
    assert elements[0].metadata['filename'] == 'book-war-and-peace-1p.txt'
    assert elements[0].metadata['text_after'] == 'used only by the elite.'

  def test_pipeline_run_chunker(self,):
    """Tests for pipeline run with chunker"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            TextPartition(chunking_strategy="by_title", max_characters=100),
            Clean_extra_whitespace(),
        ])
    elements = pipeline.run(files=TEXT_FILE_PATH)
    assert len(elements) == 38
    assert elements[0].metadata['filename'] == 'book-war-and-peace-1p.txt'
    assert elements[0].metadata['languages'] == ['eng']
