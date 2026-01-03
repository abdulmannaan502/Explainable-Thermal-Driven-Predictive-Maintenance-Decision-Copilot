from llama_cpp import Llama

class MaintenanceLLM:
    def __init__(
        self,
        model_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0  # keep 0 for CPU; increase if GPU works
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    def generate(self, prompt):
        response = self.llm(
            prompt,
            max_tokens=350,
            temperature=0.2,
            top_p=0.9,
            stop=["</s>"]
        )
        return response["choices"][0]["text"].strip()
