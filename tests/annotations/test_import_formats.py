import pytest

from clarifai_datautils import ImageAnnotations
from clarifai_datautils.constants.base import DATASET_UPLOAD_TASKS
from tests.utils.annotations import get_asset_path

IMAGENET_PATH = get_asset_path('imagenet_dataset')
CIFAR_PATH = get_asset_path('cifar10_dataset')
MNIST_PATH = get_asset_path('mnist_dataset')
VGG_PATH = get_asset_path('vgg_face2_dataset')
LFW_PATH = get_asset_path('lfw_dataset')

VOC_PATH = get_asset_path('voc_dataset')
YOLO_PATH = get_asset_path('yolo_dataset')
COCO_DETECTION_PATH = get_asset_path('coco_detection')
CVAT_PATH = get_asset_path('cvat_dataset')
KITTI_PATH = get_asset_path('kitti_detection')
LABEL_ME_PATH = get_asset_path('labelme_dataset')
OPEN_IMAGES_PATH = get_asset_path('openimages_dataset')
CLARIFAI_PATH = get_asset_path('clarifai_dataset')

COCO_SEGMENTATION_PATH = get_asset_path('coco_segmentation')
CITYSCAPES_PATH = get_asset_path('cityscapes_dataset')
ADE_PATH = get_asset_path('ade20k2017_dataset')


@pytest.fixture
def annotation_object():
  return ImageAnnotations.import_from(path=IMAGENET_PATH, format='imagenet')


class Testannotaionimport:
  """Tests for annotation import.
  """

  def test_get_info(self, annotation_object):
    info = annotation_object.get_info()
    assert info['size'] == 3
    assert info['source_path'] == IMAGENET_PATH
    assert info['annotated_items_count'] == 3
    assert info['annotations_count'] == 3

  def test_detect_format(self,):
    format = ImageAnnotations.detect_format(path=IMAGENET_PATH)
    assert format == 'imagenet'

  def test_imagenet_import(self,):
    annotation_object = ImageAnnotations.import_from(path=IMAGENET_PATH, format='imagenet')
    assert annotation_object.annotation_format == 'imagenet'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION

  def test_cifar_import(self,):
    annotation_object = ImageAnnotations.import_from(path=CIFAR_PATH, format='cifar')
    assert annotation_object.annotation_format == 'cifar'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION
    assert len(annotation_object._dataset._data) == 5  # 5 images
    assert annotation_object._dataset.get_annotated_items() == 5

  def test_mnist_import(self,):
    annotation_object = ImageAnnotations.import_from(path=MNIST_PATH, format='mnist')
    assert annotation_object.annotation_format == 'mnist'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION
    assert len(annotation_object._dataset._data) == 5  # 5 images
    assert annotation_object._dataset.get_annotated_items() == 5

  def test_vgg_face2_import(self,):
    annotation_object = ImageAnnotations.import_from(path=VGG_PATH, format='vgg_face2')
    assert annotation_object.annotation_format == 'vgg_face2'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION
    assert len(annotation_object._dataset._data) == 4  # 4 images

  def test_lfw_import(self,):
    annotation_object = ImageAnnotations.import_from(path=LFW_PATH, format='lfw')
    assert annotation_object.annotation_format == 'lfw'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_CLASSIFICATION
    assert len(annotation_object._dataset._data) == 3  # 3 images

  def test_voc_detection_import(self,):
    annotation_object = ImageAnnotations.import_from(path=VOC_PATH, format='voc_detection')
    assert annotation_object.annotation_format == 'voc_detection'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 1  # 1 image

  def test_yolo_import(self,):
    annotation_object = ImageAnnotations.import_from(path=YOLO_PATH, format='yolo')
    assert annotation_object.annotation_format == 'yolo'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 1  # 1 images
    assert annotation_object._dataset.get_annotated_items() == 1
    assert annotation_object._dataset.get_annotations() == 2  # 2 annotations

  def test_coco_detection_import(self,):
    annotation_object = ImageAnnotations.import_from(
        path=COCO_DETECTION_PATH, format='coco_detection')
    assert annotation_object.annotation_format == 'coco_detection'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 2  # 2 images

  def test_cvat_import(self,):
    annotation_object = ImageAnnotations.import_from(path=CVAT_PATH, format='cvat')
    assert annotation_object.annotation_format == 'cvat'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 8  # 8 images

  def test_kitti_import(self,):
    annotation_object = ImageAnnotations.import_from(path=KITTI_PATH, format='kitti')
    assert annotation_object.annotation_format == 'kitti'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 2  # 2 images

  def test_label_me_import(self,):
    annotation_object = ImageAnnotations.import_from(path=LABEL_ME_PATH, format='label_me')
    assert annotation_object.annotation_format == 'label_me'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 1  # 1 images

  def test_open_images_import(self,):
    annotation_object = ImageAnnotations.import_from(path=OPEN_IMAGES_PATH, format='open_images')
    assert annotation_object.annotation_format == 'open_images'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 2  # 2 images

  def test_clarifai_import(self,):
    annotation_object = ImageAnnotations.import_from(path=CLARIFAI_PATH, format='clarifai')
    assert annotation_object.annotation_format == 'clarifai'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_DETECTION
    assert len(annotation_object._dataset._data) == 1  # 1 images
    assert annotation_object._dataset.get_annotations() == 2  # 2 annotations

  def test_coco_segmentation_import(self,):
    annotation_object = ImageAnnotations.import_from(
        path=COCO_SEGMENTATION_PATH, format='coco_segmentation')
    assert annotation_object.annotation_format == 'coco_segmentation'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_SEGMENTATION
    assert len(annotation_object._dataset._data) == 2  # 2 images
    assert annotation_object._dataset.get_annotations() == 8  # 8 annotations

  def test_cityscapes_import(self,):
    annotation_object = ImageAnnotations.import_from(path=CITYSCAPES_PATH, format='cityscapes')
    assert annotation_object.annotation_format == 'cityscapes'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_SEGMENTATION
    assert len(annotation_object._dataset._data) == 4  # 4 images
    assert annotation_object._dataset.get_annotations() == 8  # 8 annotations

  def test_ade20k2017_import(self,):
    annotation_object = ImageAnnotations.import_from(path=ADE_PATH, format='ade20k2017')
    assert annotation_object.annotation_format == 'ade20k2017'
    assert annotation_object.task == DATASET_UPLOAD_TASKS.VISUAL_SEGMENTATION
    assert len(annotation_object._dataset._data) == 2  # 2 images
