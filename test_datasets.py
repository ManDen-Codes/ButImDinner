import sys

with open("out.txt", "w") as log:
    log.write("step 1: yaml\n"); log.flush()
    import yaml
    log.write("step 2: datasets\n"); log.flush()
    from datasets import load_dataset
    log.write("step 3: peft\n"); log.flush()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    log.write("step 4: transformers\n"); log.flush()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    log.write("step 5: torch\n"); log.flush()
    import torch
    log.write("step 6: trl\n"); log.flush()
    from trl import SFTTrainer, SFTConfig
    log.write("all imports done\n")

print(open("out.txt").read())
