from clarifai_datautils.text.pipeline.PDF import PDFPartition
from clarifai_datautils.text.pipeline.Text import TextPartition
from clarifai_datautils.text.pipeline.cleaners import Clean_extra_whitespace, Group_broken_paragraphs
from clarifai_datautils.text.pipeline.extractors import ExtractDateTimeTz, ExtractEmailAddress

from clarifai_datautils.constants.pipeline import *

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
        return[
                PDFPartition(max_characters=5024, overlap=524),
                Clean_extra_whitespace(),
            ]

        
    def ocr_pdf_pipeline():
        return [
                PDFPartition(ocr=True),
            ]

        
    def structured_pdf_pipeline():
        return [
                PDFPartition(max_characters = 1024, overlap=None),
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
 