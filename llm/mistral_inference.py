from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

class MaintenanceLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.2,
                do_sample=False
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
