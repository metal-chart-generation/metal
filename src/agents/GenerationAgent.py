from .Agent import *
from .utils import *

__init__ = """"""

class GenerationAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("GenerationAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for generating code from a given prompt and image.
    The input of this agent is
    - generate_prompt: a string that describes the task
    - img_path: a path to the image that the code should generate
    - output_path: a path to the output directory
    - output_prefix: a prefix for the output files
    - output_suffix: a suffix for the output files
    The output of this agent is
    - raw_response: the raw response from the model
    - generated_code: the generated code
    - generated_code_path: the path to the generated code
    - combined_image_path: a path to the image that compares the input image and the generated image
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def act(self, generate_prompt, img_path, output_path, output_prefix, output_suffix):
      system_prompt = "You are tasked with generating code from the given image. NOTE: The only two external packages allowed for generated code is matplotlib and numpy."
      raw_response = self.v2t_generate(generate_prompt, img_path, system_prompt)
      output_name = output_prefix + "_" + output_suffix
      generated_code_path = output_path + output_name + ".py"
      generated_code = extract_validate_run_code(raw_response, generated_code_path.split(".")[0])
      combined_image_path = output_path + output_name + "_comparison.png"
      generated_image_path = output_path + output_name + ".png"
      combine_images(img_path, generated_image_path, combined_image_path)
      return raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path