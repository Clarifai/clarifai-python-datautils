from dataclasses import dataclass


@dataclass
class DATASET_UPLOAD_TASKS:
  VISUAL_CLASSIFICATION: str = "visual_classification"
  VISUAL_DETECTION: str = "visual_detection"
  VISUAL_SEGMENTATION: str = "visual_segmentation"
  TEXT_CLASSIFICATION: str = "text_classification"
  MULTIMODAL_CLASSIFICATION: str = "multimodal_upload"
