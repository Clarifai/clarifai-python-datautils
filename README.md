![Clarifai logo](docs/logo.png)

# Clarifai Python Data Utils


[![Discord](https://img.shields.io/discord/1145701543228735582)](https://discord.gg/M32V7a7a)
[![codecov](https://img.shields.io/pypi/dm/clarifai)](https://pypi.org/project/clarifai)

This is a collection of utilities for handling various types of multimedia data. Enhance your experience by seamlessly integrating these utilities with the Clarifai Python SDK. This powerful combination empowers you to address both visual and textual use cases effortlessly through the capabilities of Artificial Intelligence. Unlock new possibilities and elevate your projects with the synergy of versatile data utilities and the robust features offered by the [Clarifai Python SDK](https://github.com/Clarifai/clarifai-python). Explore the fusion of these tools to amplify the intelligence in your applications! 🌐🚀

[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)

---
## Table Of Contents

* **[Installation](#installation)**
* **[Getting Started](#getting-started)**
* **[Features](#features)**
  * [Image Utils](#image-utils)
* **[Usage](#usage)**
* **[Examples](#more-examples)**


## Installation


Install from PyPi:

```bash
pip install clarifai-utils
```

Install from Source:

```bash
git clone https://github.com/Clarifai/clarifai-python-datautils
cd clarifai-python-datautils
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```


## Getting started

Quick intro to Image Annotation Conversion feature

```python
from clarifai_utils import Image_Annotations

annotated_dataset = Image_Annotations.import_from(path= 'folder_path', format= 'annotation_format')
```

## Features

### Image Utils
- #### Annotation Conversion
  - Load various annotated image datasets and export to clarifai Platform
  - Convert from one annotation format to other supported annotation formats



## Usage
### Image Annotation conversion
```python
from clarifai_utils import Image_Annotations
#import from folder
coco_dataset = Image_Annotations.import_from(path='folder_path',format= 'coco_detection')

#clarifai dataset loader object
coco_dataset.clarifai_loader()


#info about loaded dataset
coco_dataset.get_info()


#exporting to other formats
coco_dataset.export_to('voc_detection')
```

## More Examples

See many more code examples in this [repo](https://github.com/Clarifai/examples).
