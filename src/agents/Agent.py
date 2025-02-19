from abc import ABC, abstractmethod
from .utils import *

import warnings
warnings.filterwarnings("ignore")

class Agent(ABC):
  def __init__(self, name, model_name, **kwargs):
    self.name = name
    self.model_name = model_name
    self.config = kwargs
    if name == "EvaluationAgent" or name == "VerificationAgent":
      self.model = None
    else:
      self.model = self._load_model(model_name)
    print(f"{name} loaded.")
    
    
  def _load_model(self, model_name):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = self.config.get("cuda_device", "0")

    if model_name == "gpt-4o":
      from openai import OpenAI
      import dotenv
      dotenv.load_dotenv()
      OPENAI_ORG = os.getenv('OPENAI_ORG')
      OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
      client = OpenAI(api_key=OPENAI_API_KEY)
      return client
    
    elif model_name == "llama3_2":
      import torch
      from transformers import MllamaForConditionalGeneration, AutoProcessor
      model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
      model = MllamaForConditionalGeneration.from_pretrained(
      model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
      )
      processor = AutoProcessor.from_pretrained(model_id)
      return model, processor
    else:
      return None
    
  def t2t_generate(self, user_prompt, system_prompt="You are a helpful assistant."):
    
    model_name = self.model_name
    
    if model_name == "gpt-4o":
      model = self.model
      response = model.chat.completions.create(
          model=model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}
          ],
          temperature=0.7
      )
      return response.choices[0].message.content.strip()
    
    elif model_name == "llama3_2":
      model, processor = self.model
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": [
              {"type": "text", "text": user_prompt}
          ]}
      ]
      input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
      inputs = processor(
          None,
          input_text,
          add_special_tokens=False,
          return_tensors="pt"
          ).to(model.device)

      output = model.generate(**inputs, max_new_tokens=1000)
      response = processor.decode(output[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
      return response.strip()
    else:
      return None
      
  def v2t_generate(self, user_prompt, image_path, system_prompt="You are a helpful assistant."):
      import base64
      
      with open(image_path, "rb") as image_file:
        image_url = base64.b64encode(image_file.read()).decode('utf-8')
      model_name = self.model_name
      
      if model_name == "gpt-4o":
        model = self.model
        response = model.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                      {"type": "text",
                       "text": user_prompt},
                      {
                          "type": "image_url",
                          "image_url": {
                              "url": f"data:image/jpeg;base64,{image_url}", },
                      },
                  ]},
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
      elif model_name == "llama3_2":
        model, processor = self.model
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": [
              {"type": "image"},
              {"type": "text", "text": user_prompt}
          ]}
       ]
      input_img = Image.open(image_path)
      input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
      inputs = processor(
          input_img,
          input_text,
          add_special_tokens=False,
          return_tensors="pt"
      ).to(model.device)
      output = model.generate(**inputs, max_new_tokens=1000)
      response = processor.decode(output[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
      return response.strip()
  
  @abstractmethod
  def act(self, *args, **kwargs):
    pass