from .Agent import *
from .utils import *

import warnings
warnings.filterwarnings("ignore")

__init__ = """"""

class EvaluationAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("EvaluationAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for evaluating the generated code.
    The input of this agent is
    - golden_code_file: a path to the golden code file
    - generated_code_file: a path to the generated code file
    The output of this agent is
    - results: a dictionary containing the evaluation results
    - revision_rank: a list of the keys in the results dictionary sorted by the values in the results dictionary from the lowest to the highest
    """
    for line in info.split("\n"):
      print("|", line)
    return info
    
  def act(self, golden_code_file, generated_code_file):
    from packages.ChartMimic.chart2code.utils.evaluator.text_evaluator import TextEvaluator
    from packages.ChartMimic.chart2code.utils.evaluator.chart_type_evaluator import ChartTypeEvaluator
    from packages.ChartMimic.chart2code.utils.evaluator.color_evaluator import ColorEvaluator
    from packages.ChartMimic.chart2code.utils.evaluator.layout_evaluator import LayoutEvaluator
    
    text_evaluator = TextEvaluator(use_position=False, use_axs=True)
    type_evaluator = ChartTypeEvaluator()
    color_evaluator = ColorEvaluator()
    layout_evaluator = LayoutEvaluator()
    
    text_evaluator(generated_code_file, golden_code_file)
    type_evaluator(generated_code_file, golden_code_file)
    color_evaluator(generated_code_file, golden_code_file)
    layout_evaluator(generated_code_file, golden_code_file)
    results = {
        "text": text_evaluator.metrics["f1"],
        "type": type_evaluator.metrics["f1"],
        "color": color_evaluator.metrics["f1"],
        "layout": layout_evaluator.metrics["f1"]
    }
    
    file_name = os.path.basename(generated_code_file)
    intermediate_pdf = file_name.replace('.py', '.pdf')
    intermediate_img = file_name.replace('.py', '.png')
    if os.path.exists(intermediate_pdf):
        os.remove(intermediate_pdf)
    if os.path.exists(intermediate_img):
        os.remove(intermediate_img)
        
    working_dir = '/'.join(generated_code_file.split('/')[:-1])
    trash_pdf = f"{working_dir}.pdf"
    trash_png = f"{working_dir}.png"
    if os.path.exists(trash_pdf):
        os.remove(trash_pdf)
    if os.path.exists(trash_png):
        os.remove(trash_png)
    
    revision_rank = []
    revision_target = [k for k, v in results.items() if v != 1]
    if len(revision_target) > 0:
        revision_rank = sorted(revision_target, key=lambda x: results[x])
    return results, revision_rank
    
      
