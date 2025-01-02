from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

save_lora="lora_checkpoint2"
save_merge="merged"

model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("sh2orc/Llama-3.1-Korean-8B-Instruct", device_map="auto"),\
                                            save_lora)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(save_merge)
