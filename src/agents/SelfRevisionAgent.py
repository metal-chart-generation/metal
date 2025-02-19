from .Agent import *
from .utils import *

__init__ = """"""

class SelfRevisionAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("SelfRevisionAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for revising the generated code.
    The input of this agent is
    - generated_code: the generated code
    - output_path: a path to the output directory
    - output_prefix: a prefix for the output files
    - output_suffix: a suffix for the output files
    The output of this agent is
    - prompt: the prompt for the revision
    - raw_response: the raw response from the model
    - revised_code: the revised code
    - revised_code_path: the path to the revised code
    - revised_img_path: a path to the image that the revised code generates
    - revised_pdf_path: a path to the pdf that the revised code generates
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def self_revision_prompt_assembler(self, code):
    from agents.prompts.self_revision_prompt import self_revision_prompt_template
    prompt = self_revision_prompt_template.format(code=code)
    return prompt
  
  def act(self, groundtruth_img_path, generated_code, output_path, output_prefix, output_suffix):
    prompt = self.self_revision_prompt_assembler(generated_code)
    raw_response = self.v2t_generate(prompt, groundtruth_img_path)
    
    revised_code = raw_response.split("```python")[1].split("```")[0].strip()
    output_name = output_prefix + "_" + output_suffix
    revised_code_path = output_path + output_name + ".py"
    revised_img_path = output_path + output_name + ".png"
    revised_pdf_path = output_path + output_name + ".pdf"
        
    run_code_generate(revised_code, revised_code_path.split(".")[0])
    revised_combined_image_path = output_path + output_name + "_comparison.png"
    combine_images(groundtruth_img_path, output_path + output_name + ".png", revised_combined_image_path)
        
    return prompt, raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path
  
  