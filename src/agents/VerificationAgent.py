from .Agent import *
from .utils import *

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import easyocr

__init__ = """"""

class VerificationAgent(Agent):
  def __init__(self, model, **kwargs):
    ocr_model_path = os.environ.get("EASYOCR_MODEL_PATH", "./easyocr_model")
    self.reader = easyocr.Reader(['en'], model_storage_directory=ocr_model_path)
    super().__init__("VerificationAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for compute the similarity between the golden cahrt and the generated chart.
    The input of this agent is
    - golden_img: The path to the golden chart image
    - generated_img: The path to the generated chart image
    The output of this agent is
    - results: a dictionary containing the evaluation results
    - revision_rank: a list of the keys in the results dictionary sorted by the values in the results dictionary from the lowest to the highest
    """
    for line in info.split("\n"):
      print("|", line)
    return info
  
  def extract_text(self, image_path):
        results = self.reader.readtext(image_path)
        return [text[1] for text in results]
    
  def text_similarity(self, golden_img, generated_img):
      text1 = self.extract_text(golden_img)
      text2 = self.extract_text(generated_img)
      
      intersection = len(set(text1).intersection(set(text2)))
      union = len(set(text1).union(set(text2)))
      return intersection / union if union > 0 else 0
    
  
  def extract_colors(self, image_path, top_n=20):
      color_ranges = {
        "red1": [(0, 100, 100), (10, 255, 255)],  # First red hue range
        "red2": [(170, 100, 100), (180, 255, 255)],  # Second red hue range
        "green": [(35, 50, 50), (85, 255, 255)],  # Green hue range
        "blue": [(100, 50, 50), (140, 255, 255)],  # Blue hue range
        "orange": [(10, 100, 100), (25, 255, 255)],  # Orange hue range
        "yellow": [(25, 100, 100), (35, 255, 255)],  # Yellow hue range
        "cyan": [(85, 100, 100), (100, 255, 255)],  # Cyan hue range
        "magenta": [(140, 100, 100), (170, 255, 255)],  # Magenta hue range
        "purple": [(125, 100, 50), (150, 255, 255)],  # Purple hue range
        "brown": [(10, 50, 20), (30, 255, 200)],  # Merged Brown & Beige range
        "pink": [(150, 100, 100), (170, 255, 255)],  # Pink hue range
        "light_blue": [(90, 50, 100), (110, 255, 255)],  # Light blue range
        "dark_blue": [(100, 150, 50), (130, 255, 150)],  # Dark blue range
        "dark_green": [(35, 100, 20), (85, 255, 120)],  # Dark green hue range
        "lime": [(40, 100, 100), (70, 255, 255)],  # Lime hue range
        "teal": [(80, 100, 100), (100, 255, 255)],  # Teal hue range
        "olive": [(30, 50, 50), (40, 255, 150)],  # Olive hue range
        "black_gray": [(0, 0, 0), (180, 50, 200)],  # Merged Black & Gray range
        "white": [(0, 0, 200), (180, 50, 255)]  # White range
     }
    
      image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
      if image.shape[2] == 3:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
      hsv_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
      
      color_counter = Counter()
      for color_name, (lower, upper) in color_ranges.items():
          lower = np.array(lower, dtype="uint8")
          upper = np.array(upper, dtype="uint8")
          mask = cv2.inRange(hsv_image, lower, upper)
          color_counter[color_name] = cv2.countNonZero(mask)
          
      return color_counter

  
  def color_similarity(self, golden_img, generated_img):
      colors1 = self.extract_colors(golden_img)
      colors2 = self.extract_colors(generated_img)
      
      all_colors = set(colors1.keys()).union(set(colors2.keys()))
      vec1 = np.array([colors1.get(c, 0) for c in all_colors]).reshape(1, -1)
      vec2 = np.array([colors2.get(c, 0) for c in all_colors]).reshape(1, -1)
      
      return cosine_similarity(vec1, vec2)[0, 0]

  def overall_similarity(self, golden_img, generated_img):
      img1 = cv2.imread(golden_img, cv2.IMREAD_GRAYSCALE)
      img2 = cv2.imread(generated_img, cv2.IMREAD_GRAYSCALE)
      
      img1 = cv2.resize(img1, (300, 300))
      img2 = cv2.resize(img2, (300, 300))
      
      return ssim(img1, img2)
    
  def act(self, golden_img, generated_img):
    
    text_similarity = self.text_similarity(golden_img, generated_img)
    color_similarity = self.color_similarity(golden_img, generated_img)
    overall_similarity = self.overall_similarity(golden_img, generated_img)
    
    results = {
        "text": text_similarity,
        "color": color_similarity,
        "ouverall": overall_similarity
    }
    
    revision_rank = []
    revision_target = [k for k, v in results.items() if v != 1]
    if len(revision_target) > 0:
        revision_rank = sorted(revision_target, key=lambda x: results[x])
    return results, revision_rank
    
      
