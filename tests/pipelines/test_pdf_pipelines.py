import os.path as osp

from clarifai_datautils.multimodal import PDFPartition, Pipeline
from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
from clarifai_datautils.multimodal.pipeline.extractors import ExtractTextAfter

PDF_FILE_PATH = osp.abspath(osp.join(osp.dirname(__file__), "assets", "DA-1p.pdf"))


class TestPDFPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartition(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractTextAfter(key='text_after', string='demon to survive')
        ])
    elements = pipeline.run(files=PDF_FILE_PATH)
    assert len(elements) == 3
    assert elements[0].text[:9] == 'MAIN GAME'
    assert elements[0].metadata['filename'] == 'DA-1p.pdf'
    assert elements[0].metadata['page_number'] == 1
    assert elements[0].metadata['text_after'] == 'our assault."'

  def test_pipeline_run_chunker(self,):
    """Tests for pipeline run with chunker"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartition(chunking_strategy="by_title", max_characters=10),
            Clean_extra_whitespace(),
        ])
    elements = pipeline.run(files=PDF_FILE_PATH)
    assert len(elements) == 315
    assert elements[0].text[:9] == 'MAIN GAME'
    assert elements[0].metadata['filename'] == 'DA-1p.pdf'
    assert elements[0].metadata['page_number'] == 1

  def test_pipeline_run_chunker_overlap(self,):
    """Tests for pipeline run with chunker overlap"""
    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartition(
                chunking_strategy="by_title", max_characters=1024, overlap=20, overlap_all=True),
            Clean_extra_whitespace(),
        ])
    elements = pipeline.run(files=PDF_FILE_PATH)
    assert len(elements) == 3
    assert elements[0].text[:9] == 'MAIN GAME'
    assert elements[0].metadata['filename'] == 'DA-1p.pdf'
    assert elements[0].metadata['page_number'] == 1
    assert elements[0].text[-20:] == 'urvive our assault."'
    assert elements[1].text[:20] == 'urvive our assault."'

  def test_pipeline_run_ocr(self,):
    """Tests for pipeline run with chunker overlap"""
    pipeline = Pipeline(
        name='pipeline-ocr', transformations=[
            PDFPartition(max_characters=1024, ocr=True),
        ])
    assert pipeline.transformations[0].strategy == 'ocr_only'
