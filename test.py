import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

swanlab.init(project="medical-assistant",
             experiment_name="medical-assistant")

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

swanlab.log({"Prediction": 2})

swanlab.finish()
