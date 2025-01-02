import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from datasets import Dataset,DatasetDict
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

base_model_id = "Llama-3.1-Korean-8B-Instruct"
# dataset_name = "scooterman/guanaco-llama3-1k"


ds3 = load_dataset("kyujinpy/OpenOrca-KO", split="train")

main_list=[]
main_list2=[]
for i in range(len(ds3)):
    
    dict={}
    factor=ds3[i]["input"]
    factor2=ds3[i]["instruction"]
    factor3=ds3[i]["output"]
    A=f"{factor} {factor2}"
    B=f"{factor3}"

    word1=f"{A}"
    word2=f"{B}"

    dict["text"]=f"""<|start_header_id|>user<|end_header_id|>{{{word1}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{{word2}}}<|eot_id|>"""
    if i<16000:
        main_list.append(dict)
    else:
        main_list2.append(dict)

new_dataset=Dataset.from_list(main_list)
dataset=new_dataset

new_dataset2=Dataset.from_list(main_list2)
dataset2=new_dataset2

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(
    "Llama-3.1-Korean-8B-Instruct",
    add_eos_token=True,
    add_bos_token=True, 
)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "results4"
num_train_epochs = 1000
bf16 = True
per_device_train_batch_size = 1
gradient_accumulation_steps = 12
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
#max_steps = 50000
warmup_ratio = 0.03
group_by_length = True
save_steps = 200
logging_steps = 1
save_total_limit=3
save_lora="lora_checkpoint2"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    report_to="wandb",

    evaluation_strategy="epoch",
    per_device_eval_batch_size=1,  
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset2,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.save_model(output_dir)

model.save_pretrained(save_lora, save_adapter=True, save_config=True)
# del model
# model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_id).to("cuda"), "lora_checkpoint")

# merged_model = model_to_merge.merge_and_unload()
# merged_model.save_pretrained(merged_model)


# A={'text': '<|start_header_id|>user<|end_header_id|>\
#    {{Me gradué hace poco de la carrera de medicina \
#    ¿Me podrías aconsejar para conseguir rápidamente \
#    un puesto de trabajo?}}\
#    <|eot_id|>\
#    <|start_header_id|>\
#    assistant<|end_header_id|>\
#    {{Esto vale tanto para \
#    médicos como para cualquier otra profesión tras \
#    finalizar los estudios aniversarios y mi consejo\
#     sería preguntar a cuántas personas haya conocido\
#     mejor. En este caso, mi primera opción sería hablar \
#    con otros profesionales médicos, echar currículos \
#    en hospitales y cualquier centro de salud. En paralelo,\
#     trabajaría por mejorar mi marca personal como \
#    médico mediante un blog o formas digitales de \
#    comunicación como los vídeos. Y, para mejorar \
#    las posibilidades de encontrar trabajo, también\
#     participaría en congresos y encuentros para conseguir más contactos. \
#    Y, además de todo lo anterior, seguiría estudiando para \
#    presentarme a las oposiciones y ejercer la medicina en el \
#    sector público de mi país.}}\
#    <|eot_id|>'}