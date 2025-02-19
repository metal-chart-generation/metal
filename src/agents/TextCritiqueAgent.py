from .Agent import *
from .utils import *

__init__ = """"""

class TextCritiqueAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("TextCritiqueAgent", model, **kwargs)
  
  def get_agent_info(self):
    info = """This agent is responsible for critiquing the generated code based on the critique.
    The input of this agent is
    - critique: a string that describes the critique
    - code: the generated code
    The output of this agent is
    - prompt: the prompt for the critique
    - raw_response: the raw response from the model
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def text_critique_prompt_assembler(self, critique, code):
    from agents.prompts.text_critique_prompt import critique_to_guidelines_prompt
    prompt = critique_to_guidelines_prompt.format(critique=critique, code=code)
    return prompt
    
  def act(self, critique, code):
    prompt = self.text_critique_prompt_assembler(critique, code)
    raw_response = self.t2t_generate(prompt)
    return prompt, raw_response