from .Agent import *
from .utils import *

__init__ = """"""

class TextCritiqueVisualAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("TextCritiqueVisualAgent", model, **kwargs)
  
  def get_agent_info(self):
    info = """This agent is responsible for critiquing the generated code based on the input image.
    The input of this agent is
    - groundtruth_img_path: a path to the groundtruth image
    - code: the generated code
    The output of this agent is
    - prompt: the prompt for the critique
    - raw_response: the raw response from the model
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def text_critique_prompt_assembler(self, code):
    from agents.prompts.text_only_critique_prompt import critique_to_guidelines_prompt
    prompt = critique_to_guidelines_prompt.format(code=code)
    return prompt
    
  def act(self, groundtruth_img_path, code):
    prompt = self.text_critique_prompt_assembler(code)
    raw_response = self.v2t_generate(prompt, groundtruth_img_path)
    return prompt, raw_response