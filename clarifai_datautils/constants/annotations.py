ANNOTATION_FORMATS = [
    'coco_segmentation', 'voc_detection', 'yolo', 'cifar', 'coco_detection', 'cvat', 'imagenet',
    'kitti', 'label_me', 'mnist', 'open_images', 'vgg_face2', 'lfw', 'cityscapes', 'ade20k2017'
]

ANNOTATION_TASKS = ['visual_classification', 'visual_detection', 'visual_segmentation']

ANNOTATION_FORMATS_TO_TASKS = {
    'imagenet': 'visual_classification',
    'cifar': 'visual_classification',
    'mnist': 'visual_classification',
    'vgg_face2': 'visual_classification',
    'lfw': 'visual_classification',
    'voc_detection': 'visual_detection',
    'yolo': 'visual_detection',
    'coco_detection': 'visual_detection',
    'cvat': 'visual_detection',
    'kitti': 'visual_detection',
    'label_me': 'visual_detection',
    'open_images': 'visual_detection',
    'coco_segmentation': 'visual_segmentation',
    'cityscapes': 'visual_segmentation',
    'ade20k2017': 'visual_segmentation',
}

FORMAT_MAP = {
    'coco_segmentation': 'coco',
    'voc_detection': 'voc_detection',
    'yolo': 'yolo',
    'cifar': 'cifar',
    'coco_detection': 'coco_instances',
    'cvat': 'cvat',
    'imagenet': 'imagenet',
    'kitti': 'kitti',
    'label_me': 'label_me',
    'mnist': 'mnist',
    'open_images': 'open_images',
    'vgg_face2': 'vgg_face2',
    'lfw': 'lfw',
    'cityscapes': 'cityscapes',
    'ade20k2017': 'ade20k2017',
}
