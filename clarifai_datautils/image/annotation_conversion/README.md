# Annotation Loader

A framework to load,export and analyze different annotated datasets


## Usage

### Features
```python
from clarifai_datautils import ImageAnnotations
#import from folder
coco_dataset = ImageAnnotations.import_from(path='folder_path',format= 'coco_detection')


#info about loaded dataset
coco_dataset.get_info()


#exporting to other formats
coco_dataset.export_to('voc_detection')
```


### Upload using Clarifai Python SDK
```python
from clarifai_datautils import ImageAnnotations
coco_dataset = ImageAnnotations.import_from(path='folder_path',format= 'coco_detection')

#clarifai SDK
#export CLARIFAI_PAT={your personal access token}  # set PAT as env variable
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.upload_dataset(dataloader=coco_dataset.dataloader)

```


### Export to other formats from Clarifai Platform
```python

#clarifai SDK
#export CLARIFAI_PAT={your personal access token}  # set PAT as env variable
from clarifai.client.dataset import Dataset
dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="dataset_id")
dataset.export(save_path='output.zip',split='train')

#Extract the zip file and pass the folder to ImageAnnotations
from clarifai_datautils import ImageAnnotations
clarifai_dataset = ImageAnnotations.import_from(path='folder_path',format= 'clarifai')

#export to other formats
clarifai_dataset.export_to(path='output_path',format='coco_detection',save_images=True)

```

## Supported Formats

| Annotation format                                                                                | Format       |      TASK       |
| ------------------------------------------------------------------------------------------------ | -------      | --------------- |
| [ImageNet](http://image-net.org/)                                                                | imagenet     | classification  |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                          | cifar     | classification  |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                                                       | mnist     | classification  |
| [VGGFace2](https://github.com/ox-vgg/vgg_face2)                                                  | vgg_face2     | classification  |
| [LFW](http://vis-www.cs.umass.edu/lfw/)                                                          | lfw     | classification  |
| [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)                  | voc_detection     | detection  |
| [YOLO](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data)                         | yolo     | detection  |
| [COCO](http://cocodataset.org/#format-data)                                                      | coco_detection     | detection  |
| [CVAT](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/)                           | cvat     | detection  |
| [Kitti](http://www.cvlibs.net/datasets/kitti/index.php)                                          | kitti     | detection  |
| [LabelMe](http://labelme.csail.mit.edu/Release3.0)                                               | label_me     | detection  |
| [Open Images](https://storage.googleapis.com/openimages/web/download.html)                       | open_images     | detection  |
| [Clarifai](https://github.com/Clarifai/examples/tree/main/Data_Utils)                       | clarifai     | detection  |
| [COCO(segmentation)](http://cocodataset.org/#format-data)                                     | coco_segmentation     | segmentation  |
| [Cityscapes](https://www.cityscapes-dataset.com/)                                                | cityscapes     | segmentation  |
| [ADE](https://www.cityscapes-dataset.com/)                                                       | ade20k2017     | segmentation  |



## Resources
This tool makes use of the [Datumaro Framework](https://github.com/openvinotoolkit/datumaro)
