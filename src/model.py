import openai
import time
from transformers import pipeline

import torch

class Model:
    def __init__(self, model_name):
        if model_name == "gpt4o":
            self.model = GPT4O()
        elif model_name == "llama3":
            self.model = Llama3()
        else:
            assert False, "Invalid model name"
    
    def run(self, prompt, max_tokens=1000):
        if type(prompt) == str:
            prompt = self.model.format_prompt(prompt)
        
        response = self.model.run(prompt, max_tokens=max_tokens)
        return response
    
    def format_prompt(self, prompt):
        return self.model.format_prompt(prompt)
    
    def postprocess_response(self, response):
        return self.model.postprocess_response(response)

class GPT4O:
    def __init__(self):
        self.client = openai.OpenAI()
    
    def format_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]
    
    def run(self, prompt, max_tokens=1000):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", # model = "deployment_name"
                messages = prompt,
                temperature=0,
                n=1,
                max_tokens=max_tokens,
            )
            response = response.choices[0].message.content
        except Exception as e:
            time.sleep(1)
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", # model = "deployment_name"
                    messages = prompt,
                    temperature=0,
                    n=1,
                    max_tokens=max_tokens,
                )
                response = response.choices[0].message.content
            except:
                print("impossible to run")
                response = None
        time.sleep(1)
        return response
    
    def postprocess_response(self, response):
        return {"role": "assistant", "content": response}

class Llama3:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct", world_size=1, gpu_memory_utilization=0.95):
        self.model_id = "meta-llama/Llama-3.3-70B-Instruct"

        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    def format_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]
    
    def run(self, prompt, max_tokens=1000):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        return outputs[0]["generated_text"][-1]["content"]