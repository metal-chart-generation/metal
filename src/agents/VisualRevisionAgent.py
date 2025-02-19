from .Agent import *
from .utils import *

__init__ = """"""

class VisualRevisionAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("VisualRevisionAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for revising the generated code based on visual critique.
    The input of this agent is
    - generated_code: the generated code
    - visual_critique: a string that describes the critique
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
    
  def revision_prompt_assembler(self, code, visual_critique):
    from agents.prompts.visual_only_revision_prompt import code_revision_prompt
    prompt = code_revision_prompt.format(code=code, visual_critique=visual_critique)
    return prompt
  
  def act(self, visual_critique, groundtruth_img_path, generated_code, output_path, output_prefix, output_suffix):
    prompt = self.revision_prompt_assembler(generated_code, visual_critique)
    raw_response = self.t2t_generate(prompt)
    
    revised_code = raw_response.split("```python")[1].split("```")[0].strip()
    output_name = output_prefix + "_" + output_suffix
    revised_code_path = output_path + output_name + ".py"
    revised_img_path = output_path + output_name + ".png"
    revised_pdf_path = output_path + output_name + ".pdf"
    
    lines = revised_code.split("\n")
    lines_tb_removed = []
    for i, line in enumerate(lines):
        if "TODO" in line:
            lines_tb_removed.append(i)
    for i in sorted(lines_tb_removed, reverse=True):
        lines.pop(i)
    revised_code = "\n".join(lines)
        
    run_code_generate(revised_code, revised_code_path.split(".")[0])
    revised_combined_image_path = output_path + output_name + "_comparison.png"
    combine_images(groundtruth_img_path, output_path + output_name + ".png", revised_combined_image_path)
        
    return prompt, raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path
  
  