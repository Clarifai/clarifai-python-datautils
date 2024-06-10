import datetime

from unstructured.staging.base import dict_to_elements

from clarifai_datautils.text.pipeline.cleaners import (
    Bytes_string_to_string, Clean_bullets, Clean_dashes, Clean_extra_whitespace,
    Clean_non_ascii_chars, Clean_ordered_bullets, Clean_postfix, Clean_prefix,
    Group_broken_paragraphs, Remove_punctuation, Replace_unicode_quotes)
from clarifai_datautils.text.pipeline.extractors import (ExtractDateTimeTz, ExtractEmailAddress,
                                                         ExtractIpAddress, ExtractIpAddressName,
                                                         ExtractTextAfter, ExtractTextBefore)


class TestPipelineTransformations:
  """Tests for pipeline transformations."""

  def test_extractors_email(self,):
    extractor = ExtractEmailAddress()
    element = extractor(
        dict_to_elements([{
            "text":
                """Me me@email.com and You <You@email.com>
    ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)""",
            "type":
                "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {'email_address': ['me@email.com', 'you@email.com']}

  def test_extractors_ip(self,):
    extractor = ExtractIpAddress()
    element = extractor(
        dict_to_elements([{
            "text":
                """Me me@email.com and You <You@email.com>
    ([ba23::58b5:2236:45g2:88h2]) (10.0.2.01)""",
            "type":
                "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {
        'ip_address': ['ba23::58b5:2236:45g2:88h2', '10.0.2.01']
    }

  def test_extractors_ip_name(self,):
    extractor = ExtractIpAddressName()
    element = extractor(
        dict_to_elements([{
            "text":
                """from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by
  \n ABC.DEF.local2 ([ba23::58b5:2236:45g2:88h2%25]) with mapi id\
  n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200""",
            "type":
                "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {'ip_address_name': ['ABC.DEF.local', 'ABC.DEF.local']}

  def test_extractors_datetime(self,):
    extractor = ExtractDateTimeTz()
    element = extractor(
        dict_to_elements([{
            "text":
                """from ABC.DEF.local ([ba23::58b5:2236:45g2:88h2]) by
  \n ABC.DEF.local2 ([ba23::58b5:2236:45g2:88h2%25]) with mapi id\
  n 32.88.5467.123; Fri, 26 Mar 2021 11:04:09 +1200""",
            "type":
                "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {
        'date_time':
            datetime.datetime(
                2021, 3, 26, 11, 4, 9, tzinfo=datetime.timezone(datetime.timedelta(seconds=43200)))
    }

  def test_extractors_text_after(self,):
    extractor = ExtractTextAfter(key='text_after', string='this is the text after')
    element = extractor(
        dict_to_elements([{
            "text": """This is a test text, and this is the text after the comma""",
            "type": "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {'text_after': 'the comma'}

  def test_extractors_text_before(self,):
    extractor = ExtractTextBefore(
        key='text_before', string=', and this is the text before the comma')
    element = extractor(
        dict_to_elements([{
            "text": """This is a test text, and this is the text before the comma""",
            "type": "NarrativeText"
        }]))
    assert element[0].metadata.to_dict() == {'text_before': 'This is a test text'}

  def test_cleaners_extra_whitespace(self,):
    cleaner = Clean_extra_whitespace()
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with extra        whitespace",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with extra whitespace"

  def test_cleaners_unicode_quotes(self,):
    cleaner = Replace_unicode_quotes()
    elements = cleaner(
        dict_to_elements([{
            "text": "\x93A lovely quote!\x94",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "“A lovely quote!”"

  def test_cleaners_dashes(self,):
    cleaner = Clean_dashes()
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with - dashes",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with   dashes"

  def test_cleaners_bullets(self,):
    cleaner = Clean_bullets()
    elements = cleaner(
        dict_to_elements([{
            "text": "• This is a test text with bullets",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with bullets"

  def test_cleaners_group_broken_paragraphs(self,):
    cleaner = Group_broken_paragraphs()
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with broken\nparagraphs",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with broken paragraphs"

  def test_cleaners_remove_punctuation(self,):
    cleaner = Remove_punctuation()
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with punctuation!",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with punctuation"

  def test_cleaners_bytes_string_to_string(self,):
    cleaner = Bytes_string_to_string()
    elements = cleaner(
        dict_to_elements([{
            "text": "Hello ð\x9f\x98\x80",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "Hello 😀"

  def test_cleaners_non_ascii_chars(self,):
    cleaner = Clean_non_ascii_chars()
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with non-ascii characters: é",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with non-ascii characters: "

  def test_cleaners_ordered_bullets(self,):
    cleaner = Clean_ordered_bullets()
    elements = cleaner(
        dict_to_elements([{
            "text": "1. First bullet point, 2. Second bullet point, 3. Third bullet point",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "First bullet point, 2. Second bullet point, 3. Third bullet point"

  def test_cleaners_prefix(self,):
    cleaner = Clean_prefix(pattern='This is a test text with a prefix: ')
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with a prefix: This is the text",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is the text"

  def test_cleaners_postfix(self,):
    cleaner = Clean_postfix(pattern='This is the text')
    elements = cleaner(
        dict_to_elements([{
            "text": "This is a test text with a postfix: This is the text",
            "type": "NarrativeText"
        }]))
    assert elements[0].text == "This is a test text with a postfix:"
