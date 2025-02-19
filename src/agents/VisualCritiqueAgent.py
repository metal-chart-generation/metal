from .Agent import *
from .utils import *

__init__ = """"""

class VisualCritiqueAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("VisualCritiqueAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for critiquing the generated code based on the visual comparison.
    The input of this agent is
    - lowest_metric: the lowest metric among the evaluation results
    - combined_img_path: a path to the image that compares the input image and the generated image
    The output of this agent is
    - prompt: the prompt for the critique
    - raw_response: the raw response from the model
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def visual_critique_prompt_assembler(self, lowest_metric, combined_img_path):
    from agents.prompts.visual_critique_prompt import metric_instructions, prompts_template
    if lowest_metric == "color":
        reference_colors, generated_colors = dominant_colors_extractor(combined_img_path)
        instructions = metric_instructions[lowest_metric].format(reference_colors=reference_colors, generated_colors=generated_colors)
    else:
        instructions = metric_instructions[lowest_metric]
    prompt = prompts_template.format(lowest_metric=lowest_metric, instructions=instructions)
    return prompt
    
  def act(self, lowest_metric, combined_image_path):
    prompt = self.visual_critique_prompt_assembler(lowest_metric, combined_image_path)
    raw_response = self.v2t_generate(prompt, combined_image_path)
    return prompt, raw_response 