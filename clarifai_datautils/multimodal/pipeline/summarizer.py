import base64
import os
import random
from typing import List

try:
  from unstructured.documents.elements import CompositeElement, ElementMetadata, Image
except ImportError:
  raise ImportError(
      "Could not import unstructured package. "
      "Please install it with `pip install 'unstructured[pdf] @ git+https://github.com/clarifai/unstructured.git@support_clarifai_model'`."
  )

from clarifai.client.input import Inputs
from clarifai.client.model import Model

from .basetransform import BaseTransform


class ImageSummarizer(BaseTransform):
  """ Summarizes image elements. """

  def __init__(self, model_url="https://clarifai.com/qwen/qwen-VL/models/qwen-VL-Chat"):
    """Initializes an LlamaIndexWrapper object.

    Args:
        model_url (str): Model URL to use for summarization.

    """
    self.model_url = model_url
    self.model = Model(url=model_url, pat=os.environ.get("CLARIFAI_PAT"))
    self.summary_prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

  def __call__(self, elements: List) -> List:
    """Applies the transformation.

    Args:
        elements (List[str]): List of text elements.

    Returns:
        List of transformed text elements.

    """
    img_elements = []
    for _, element in enumerate(elements):
      element.metadata.update(
          ElementMetadata.from_dict({
              'is_original': True,
              'input_id': f'{random.randint(1000000, 99999999)}'
          }))
      if isinstance(element, Image):
        img_elements.append(element)
    # new_elements = Parallel(n_jobs=len(elements))(delayed(self._summarize_image)(element) for element in img_elements)
    new_elements = self._summarize_image(elements)
    elements.extend(new_elements)
    return elements

  def _summarize_image(self, image_elements: List[Image]) -> List[CompositeElement]:
    """Summarizes an image element.

    Args:
        image_element (Image): Image element to summarize.

    Returns:
        Summarized image element.

    """
    img_inputs = []
    for element in image_elements:
      if not isinstance(element, Image):
        continue
      new_input_id = "summarize_" + element.metadata.input_id
      input_proto = Inputs.get_multimodal_input(
          input_id=new_input_id,
          image_bytes=base64.b64decode(element.metadata.image_base64),
          raw_text=self.summary_prompt)
      img_inputs.append(input_proto)
    resp = self.model.predict(img_inputs)

    new_elements = []
    for i, element in enumerate(resp.outputs):
      summary = ""
      if image_elements[i].text:
        summary = image_elements[i].text
      summary = summary + " \n " + element.data.text.raw
      eid = image_elements[i].metadata.input_id
      meta_dict = {'source_input_id': eid, 'is_original': False}
      comp_element = CompositeElement(
          text=summary,
          metadata=ElementMetadata.from_dict(meta_dict),
          element_id="summarized_" + eid)
      new_elements.append(comp_element)

    return new_elements
