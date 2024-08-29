import os.path as osp
import pytest

PDF_FILE_PATH = osp.abspath(
    osp.join(osp.dirname(__file__), "assets", "Multimodal_sample_file.pdf"))


@pytest.mark.skip(reason="Need additional build dependencies")
class TestMultimodalPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline"""
    from clarifai_datautils.multimodal import PDFExtraction, Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFExtraction(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    from clarifai_datautils.multimodal import PDFExtraction, Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
    from clarifai_datautils.multimodal.pipeline.extractors import ExtractEmailAddress

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFExtraction(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractEmailAddress()
        ])
    elements = pipeline.run(files=PDF_FILE_PATH, loader=False)
    assert len(elements) == 2
    assert type(elements) == tuple
    assert elements[0][0].metadata.to_dict()['filename'] == 'Multimodal_sample_file.pdf'
    assert elements[0][0].metadata.to_dict()['page_number'] == 1
    assert elements[0][0].metadata.to_dict()['email_address'] == ['test_extraction@gmail.com']
    assert elements[0][6].__class__.__name__ == 'Table'
    assert elements[1][0].__class__.__name__ == 'Image'
