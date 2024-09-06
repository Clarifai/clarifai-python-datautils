import os.path as osp

from clarifai_datautils.text import Pipeline

PDF_FILE_PATH = osp.abspath(osp.join(osp.dirname(__file__), "assets", "DA-1p.pdf"))
TEXT_FILE_PATH = osp.abspath(osp.join(osp.dirname(__file__), "assets", "book-war-and-peace-1p.txt"))


class TestReadyToUsePipelines:
    """Tests for ready to use pipelines."""
    
    def test_pipeline_basic_pdf(self,):
        """Tests for basic pdf pipeline"""
        pipeline = Pipeline.load(name='basic_pdf')
        assert pipeline.name == 'basic_pdf'
        assert len(pipeline.transformations) == 1
        assert pipeline.transformations[0].__class__.__name__ == 'PDFPartition'
        
    def test_pipeline_standard_pdf(self,):
        """Tests for standard pdf pipeline"""
        pipeline = Pipeline.load(name='standard_pdf')
        assert pipeline.name == 'standard_pdf'
        assert len(pipeline.transformations) == 3
        assert pipeline.transformations[0].__class__.__name__ == 'PDFPartition'
        assert pipeline.transformations[1].__class__.__name__ == 'Clean_extra_whitespace'
        assert pipeline.transformations[2].__class__.__name__ == 'Group_broken_paragraphs'
        
    def test_pipeline_context_overlap_pdf(self,):
        """Tests for context overlap pdf pipeline"""
        pipeline = Pipeline.load(name='context_overlap_pdf')
        assert pipeline.name == 'context_overlap_pdf'
        assert len(pipeline.transformations) == 2
        assert pipeline.transformations[0].__class__.__name__ == 'PDFPartition'
        assert pipeline.transformations[1].__class__.__name__ == 'Clean_extra_whitespace'
        
    def test_pipeline_ocr_pdf(self,):
        """Tests for ocr pdf pipeline"""
        pipeline = Pipeline.load(name='ocr_pdf')
        assert pipeline.name == 'ocr_pdf'
        assert len(pipeline.transformations) == 1
        assert pipeline.transformations[0].__class__.__name__ == 'PDFPartition'
        
        
    def test_pipeline_structured_pdf(self,):
        """Tests for structured pdf pipeline"""
        pipeline = Pipeline.load(name='structured_pdf')
        assert pipeline.name == 'structured_pdf'
        assert len(pipeline.transformations) == 4
        assert pipeline.transformations[0].__class__.__name__ == 'PDFPartition'
        assert pipeline.transformations[1].__class__.__name__ == 'Clean_extra_whitespace'
        assert pipeline.transformations[2].__class__.__name__ == 'ExtractDateTimeTz'
        assert pipeline.transformations[3].__class__.__name__ == 'ExtractEmailAddress'  
        
    def test_pipeline_standard_text(self,):
        """Tests for standard text pipeline"""
        pipeline = Pipeline.load(name='standard_text')
        assert pipeline.name == 'standard_text'
        assert len(pipeline.transformations) == 3
        assert pipeline.transformations[0].__class__.__name__ == 'TextPartition'
        assert pipeline.transformations[1].__class__.__name__ == 'Clean_extra_whitespace'
        assert pipeline.transformations[2].__class__.__name__ == 'Group_broken_paragraphs'
        