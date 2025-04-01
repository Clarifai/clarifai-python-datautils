# This was taken from litellm

from clarifai_datautils.text.prompt_factory import claude_2_1_pt, llama_2_chat_pt


def test_codellama_prompt_format():
  messages = [
      {
          "role": "system",
          "content": "You are a good bot"
      },
      {
          "role": "user",
          "content": "Hey, how's it going?"
      },
  ]
  expected_prompt = "<s>[INST] <<SYS>>\nYou are a good bot\n<</SYS>>\n [/INST]\n[INST] Hey, how's it going? [/INST]\n"
  assert llama_2_chat_pt(messages) == expected_prompt


def test_claude_2_1_pt_formatting():
  # Test case: User only, should add Assistant
  messages = [{"role": "user", "content": "Hello"}]
  expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
  assert claude_2_1_pt(messages) == expected_prompt

  # Test case: System, User, and Assistant "pre-fill" sequence,
  #            Should return pre-fill
  messages = [
      {
          "role": "system",
          "content": "You are a helpful assistant."
      },
      {
          "role": "user",
          "content": 'Please return "Hello World" as a JSON object.'
      },
      {
          "role": "assistant",
          "content": "{"
      },
  ]
  expected_prompt = 'You are a helpful assistant.\n\nHuman: Please return "Hello World" as a JSON object.\n\nAssistant: {'
  assert claude_2_1_pt(messages) == expected_prompt

  # Test case: System, Assistant sequence, should insert blank Human message
  #            before Assistant pre-fill
  messages = [
      {
          "role": "system",
          "content": "You are a storyteller."
      },
      {
          "role": "assistant",
          "content": "Once upon a time, there "
      },
  ]
  expected_prompt = ("You are a storyteller.\n\nHuman: \n\nAssistant: Once upon a time, there ")
  assert claude_2_1_pt(messages) == expected_prompt

  # Test case: System, User sequence
  messages = [
      {
          "role": "system",
          "content": "System reboot"
      },
      {
          "role": "user",
          "content": "Is everything okay?"
      },
  ]
  expected_prompt = "System reboot\n\nHuman: Is everything okay?\n\nAssistant: "
  assert claude_2_1_pt(messages) == expected_prompt
