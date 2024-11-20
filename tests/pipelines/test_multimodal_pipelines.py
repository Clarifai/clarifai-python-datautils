import os.path as osp

import pytest

PDF_FILE_PATH = osp.abspath(
    osp.join(osp.dirname(__file__), "assets", "Multimodal_sample_file.pdf"))


@pytest.mark.skip(reason="Need additional build dependencies")
class TestMultimodalPipelines:
  """Tests for pipeline transformations."""

  def test_pipeline(self,):
    """Tests for pipeline"""
    from clarifai_datautils.multimodal import Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
    from clarifai_datautils.multimodal.pipeline.PDF import PDFPartitionMultimodal

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartitionMultimodal(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
        ])
    assert pipeline.name == 'pipeline-1'
    assert len(pipeline.transformations) == 2

  def test_pipeline_run(self,):
    """Tests for pipeline run"""
    from clarifai_datautils.multimodal import Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
    from clarifai_datautils.multimodal.pipeline.extractors import ExtractEmailAddress
    from clarifai_datautils.multimodal.pipeline.PDF import PDFPartitionMultimodal

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartitionMultimodal(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractEmailAddress()
        ])
    elements = pipeline.run(files=PDF_FILE_PATH, loader=False)
    assert len(elements) == 14
    assert isinstance(elements, list)
    assert elements[0].metadata.to_dict()['filename'] == 'Multimodal_sample_file.pdf'
    assert elements[0].metadata.to_dict()['page_number'] == 1
    assert elements[0].metadata.to_dict()['email_address'] == ['test_extraction@gmail.com']
    assert elements[6].__class__.__name__ == 'Table'
    assert elements[-1].__class__.__name__ == 'Image'

  def test_pipeline_run_loader(self,):
    """Tests for pipeline run with loader"""
    from clarifai_datautils.multimodal import Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
    from clarifai_datautils.multimodal.pipeline.extractors import ExtractEmailAddress
    from clarifai_datautils.multimodal.pipeline.PDF import PDFPartitionMultimodal

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartitionMultimodal(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ExtractEmailAddress()
        ])
    elements = pipeline.run(files=PDF_FILE_PATH, loader=True)
    assert elements.__class__.__name__ == 'MultiModalLoader'
    assert len(elements) == 14
    assert elements.elements[0].metadata.to_dict()['filename'] == 'Multimodal_sample_file.pdf'

  def test_pipeline_summarize(self,):
    """Tests for pipeline run with summarizer"""
    from clarifai_datautils.multimodal import Pipeline
    from clarifai_datautils.multimodal.pipeline.cleaners import Clean_extra_whitespace
    from clarifai_datautils.multimodal.pipeline.PDF import PDFPartitionMultimodal
    from clarifai_datautils.multimodal.pipeline.summarizer import ImageSummarizer

    pipeline = Pipeline(
        name='pipeline-1',
        transformations=[
            PDFPartitionMultimodal(chunking_strategy="by_title", max_characters=1024),
            Clean_extra_whitespace(),
            ImageSummarizer()
        ])
    elements = pipeline.run(files=PDF_FILE_PATH, loader=False)
    assert len(elements) == 15
    assert isinstance(elements, list)
    assert elements[0].metadata.to_dict()['filename'] == 'Multimodal_sample_file.pdf'
    assert elements[0].metadata.to_dict()['page_number'] == 1
    assert elements[0].metadata.to_dict()['email_address'] == ['test_extraction@gmail.com']
    assert elements[6].__class__.__name__ == 'Table'
    assert elements[-2].__class__.__name__ == 'Image'
    assert elements[-2].metadata.is_original == True
    assert elements[-2].metadata.input_id is not None
    id = elements[-2].metadata.input_id
    assert elements[-1].__class__.__name__ == 'CompositeElement'
    assert elements[-1].metadata.is_original == False
    assert elements[-1].metadata.source_input_id == 'summarized_' + id
