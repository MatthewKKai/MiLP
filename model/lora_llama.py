from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
# from transformers import Trainer, TrainingArguments

llama_path = r"./llama-7b"
# llama = LlamaForCausalLM.from_pretrained(llama_path)
# lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
#                         inference_mode=False, 
#                         r=8, lora_alpha=32, 
#                         lora_dropout=0.1)
# lora_llama = get_peft_model(llama, lora_config)
# lora_llama.print_trainable_parameters()

class lora_llama():
    def __init__(self, base_model, config: str="lora"):
        self.base = base_model
        if config=="lora":
            self.config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=10, lora_alpha=32,
                        lora_dropout=0.1)
        else:
            print("Sorry, we only support LoRA currently")
            self.config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=8, lora_alpha=32, 
                        lora_dropout=0.1)
        self.lora_llama = get_peft_model(self.base, self.config)

    def get_lora_llama(self):
        return self.lora_llama

    # def trainable_paramters(self):
    #     print(self.lora_llama.print_trainable_parameters())


