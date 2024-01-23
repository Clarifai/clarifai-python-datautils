from clarifai_utils import \
    Image_Annotations  # change this to 'from clarifai-utils import Image_Annotations'
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

COCO_SEGMENTATION_PATH = get_asset_path('coco_segmentation')
CITYSCAPES_PATH = get_asset_path('cityscapes_dataset')
ADE_PATH = get_asset_path('ade20k2017_dataset')


class Testclarifailoader:
  """Tests for annotation object to clarifai loaders.
  """

  def test_imagenet_loader(self,):
    annotation_object = Image_Annotations.import_from(path=IMAGENET_PATH, format='imagenet')
    clarifai_loader = annotation_object.clarifai_loader()
    assert len(clarifai_loader) == 3
    assert clarifai_loader.task == 'visual_classification'
    assert clarifai_loader[0].labels[0] in ['label_0', 'lable_1']
    assert clarifai_loader[0].id == '1'

  def test_cifar_loader(self,):
    annotation_object = Image_Annotations.import_from(path=CIFAR_PATH, format='cifar')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_classification'
    assert len(clarifai_loader) == 5
    assert clarifai_loader[0].labels[0] in ['airplane', 'automobile']

  def test_mnist_loader(self,):
    annotation_object = Image_Annotations.import_from(path=MNIST_PATH, format='mnist')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_classification'
    assert len(clarifai_loader) == 5

  def test_vgg_face2_loader(self,):
    annotation_object = Image_Annotations.import_from(path=VGG_PATH, format='vgg_face2')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_classification'
    assert len(clarifai_loader) == 4

  def test_lfw_loader(self,):
    annotation_object = Image_Annotations.import_from(path=LFW_PATH, format='lfw')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_classification'
    assert len(clarifai_loader) == 3
    assert clarifai_loader[0].labels == ['name1']
    assert clarifai_loader[0].id == '0001'

  def test_voc_detection_loader(self,):
    annotation_object = Image_Annotations.import_from(path=VOC_PATH, format='voc_detection')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 1
    assert clarifai_loader[0].labels == ['cat']
    assert clarifai_loader[0].id == '000001'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_yolo_loader(self,):
    annotation_object = Image_Annotations.import_from(path=YOLO_PATH, format='yolo')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 1
    assert clarifai_loader[0].labels == ['label_2', 'label_4']
    assert clarifai_loader[0].id == '1'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_coco_detection_loader(self,):
    annotation_object = Image_Annotations.import_from(
        path=COCO_DETECTION_PATH, format='coco_detection')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 2
    assert clarifai_loader[0].labels == ['b']
    assert clarifai_loader[0].id == 'a'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_cvat_loader(self,):
    annotation_object = Image_Annotations.import_from(path=CVAT_PATH, format='cvat')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 8
    assert clarifai_loader[0].labels == ['label1']
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_kitti_loader(self,):
    annotation_object = Image_Annotations.import_from(path=KITTI_PATH, format='kitti')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 2
    assert clarifai_loader[0].labels == ['truck', 'van']
    assert clarifai_loader[0].id == '10'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_label_me_loader(self,):
    annotation_object = Image_Annotations.import_from(path=LABEL_ME_PATH, format='label_me')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 1
    assert clarifai_loader[0].labels == ['b1']
    assert clarifai_loader[0].id == 'img1'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_open_images_loader(self,):
    annotation_object = Image_Annotations.import_from(path=OPEN_IMAGES_PATH, format='open_images')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_detection'
    assert len(clarifai_loader) == 2
    assert clarifai_loader[1].id == 'aa'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_coco_segmentation_loader(self,):
    annotation_object = Image_Annotations.import_from(
        path=COCO_SEGMENTATION_PATH, format='coco_segmentation')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_segmentation'
    assert len(clarifai_loader) == 2
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_cityscapes_loader(self,):
    annotation_object = Image_Annotations.import_from(path=CITYSCAPES_PATH, format='cityscapes')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_segmentation'
    assert len(clarifai_loader) == 4
    assert isinstance(clarifai_loader[0].image_bytes, bytes)

  def test_ade20k2017_loader(self,):
    annotation_object = Image_Annotations.import_from(path=ADE_PATH, format='ade20k2017')
    clarifai_loader = annotation_object.clarifai_loader()
    assert clarifai_loader.task == 'visual_segmentation'
    assert len(clarifai_loader) == 2
    assert clarifai_loader[0].id == '1'
    assert isinstance(clarifai_loader[0].image_bytes, bytes)
