# %% [markdown]
# # A guide to fine-tuning Code Llama
# 
# **In this guide I show you how to fine-tune Code Llama to become a beast of an SQL developer. For coding tasks, you can generally get much better performance out of Code Llama than Llama 2, especially when you specialise the model on a particular task:**
# 
# - I use the [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) which is a bunch of text queries and their corresponding SQL queries
# - A Lora approach, quantizing the base model to int 8, freezing its weights and only training an adapter
# - Much of the code is borrowed from [alpaca-lora](https://github.com/tloen/alpaca-lora), but I refactored it quite a bit for this
# 

# %% [markdown]
# ### 2. Pip installs
# 

# %%
# !pip install git+https://github.com/huggingface/transformers.git@main bitsandbytes  # we need latest transformers for this
# !pip install git+https://github.com/huggingface/peft.git@4c611f4
# !pip install datasets==2.10.1
# import locale # colab workaround
# locale.getpreferredencoding = lambda: "UTF-8" # colab workaround
# !pip install wandb
# !pip install scipy

# %% [markdown]
# I used an A100 GPU machine with Python 3.10 and cuda 11.8 to run this notebook. It took about an hour to run.

# %% [markdown]
# ### Loading libraries
# 

# %%
from datetime import datetime
import os
import sys

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig


# %% [markdown]
# (If you have import errors, try restarting your Jupyter kernel)
# 

# %% [markdown]
# ### Load dataset
# 

# %%
from datasets import load_dataset
train_dataset = load_dataset('json', data_files='/home/sam/text-sql-with-execution-evaluation/spider-create-context-intersect/spider_create_context_train_db_id.json', split='train')
eval_dataset = load_dataset('json', data_files='/home/sam/text-sql-with-execution-evaluation/spider-create-context-intersect/spider_create_context_val_db_id.json', split='train')

# %% [markdown]
# The above pulls the dataset from the Huggingface Hub and splits 10% of it into an evaluation set to check how well the model is doing through training. If you want to load your own dataset do this:
# 
# ```
# train_dataset = load_dataset('json', data_files='train_set.jsonl', split='train')
# eval_dataset = load_dataset('json', data_files='validation_set.jsonl', split='train')
# ```
# 
# And if you want to view any samples in the dataset just do something like:``` ```
# 

# %%
print(train_dataset[3])

# %% [markdown]
# Each entry is made up of a text 'question', the sql table 'context' and the 'answer'.

# %% [markdown]
# ### Load model
# I load code llama from huggingface in int8. Standard for Lora:

import wandb

wandb.login(key="07d2fe3d77f6ad7bac5fb71f8154573c994f5120")

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = "codellama/CodeLlama-34b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    # torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-34b-hf")

# %% [markdown]
# torch_dtype=torch.float16 means computations are performed using a float16 representation, even though the values themselves are 8 bit ints.
# 
# If you get error "ValueError: Tokenizer class CodeLlamaTokenizer does not exist or is not currently imported." Make sure you have transformers version is 4.33.0.dev0 and accelerate is >=0.20.3.
# 

# %% [markdown]
# ### 3. Check base model
# A very good common practice is to check whether a model can already do the task at hand. Fine-tuning is something you want to try to avoid at all cost:
# 

# %%
eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# %% [markdown]
# I get the output:
# ```
# SELECT * FROM table_name_12 WHERE class > 91.5 AND city_of_license = 'hyannis, nebraska'
# ```
# which is clearly wrong if the input is asking for just class!

# %% [markdown]
# ### 4. Tokenization
# Setup some tokenization settings like left padding because it makes [training use less memory](https://ai.stackexchange.com/questions/41485/while-fine-tuning-a-decoder-only-llm-like-llama-on-chat-dataset-what-kind-of-pa):

# %%
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# %% [markdown]
# Setup the tokenize function to make labels and input_ids the same. This is basically what [self-supervised fine-tuning](https://neptune.ai/blog/self-supervised-learning) is:

# %%
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

# %% [markdown]
# And run convert each data_point into a prompt that I found online that works quite well:

# %%
def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)

# %% [markdown]
# Reformat to prompt and tokenize each sample:

# %%
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# %% [markdown]
# ### 5. Setup Lora

# %%
model.train() # put model back into training mode
model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'],
#     target_modules=[
#     "q_proj",
#     "k_proj",
#     "v_proj",
#     "o_proj",
# ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# %% [markdown]
# To resume from a checkpoint, set resume_from_checkpoint to the path of the adapter_model.bin you want to resume from. This code'll replace the lora adapter attached to the model:

# %%
resume_from_checkpoint = "" # set this to the adapter_model.bin file you want to resume from

if resume_from_checkpoint:
    if os.path.exists(resume_from_checkpoint):
        print(f"Restarting from {resume_from_checkpoint}")
        adapters_weights = torch.load(resume_from_checkpoint)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {resume_from_checkpoint} not found")

# %% [markdown]
# Optional stuff to setup Weights and Biases to view training graphs:

# %%
wandb_project = "qloracodellama34b-64batch-r8-alllinear-1e4learningrate"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


# %%
if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

# %% [markdown]
# ### 6. Training arguments
# If you run out of GPU memory, change per_device_train_batch_size. The gradient_accumulation_steps variable should ensure this doesn't affect batch dynamics during the training run. All the other variables are standard stuff that I wouldn't recommend messing with:

# %%
batch_size = 64
per_device_train_batch_size = 8
gradient_accumulation_steps = batch_size // per_device_train_batch_size
output_dir = "qloracodellama34b-64batch-r8-alllinear-1e4learningrate"

training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=400,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        output_dir=output_dir,
        # save_total_limit=3,
        load_best_model_at_end=False,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

# %% [markdown]
# Then we do some pytorch-related optimisation (which just make training faster but don't affect accuracy):

# %%
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)
if torch.__version__ >= "2" and sys.platform != "win32":
    print("compiling the model")
    model = torch.compile(model)

# %%
trainer.train()

# %% [markdown]
# ### Load the final checkpoint
# Now for the moment of truth! Has our work paid off...?

# %%
# import torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# base_model = "codellama/CodeLlama-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# # %% [markdown]
# # To load a fine-tuned Lora/Qlora adapter use PeftModel.from_pretrained. ```output_dir``` should be something containing an adapter_config.json and adapter_model.bin:

# # %%
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, output_dir)

# %% [markdown]
# Try the same prompt as before:

# %%
# eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

# You must output the SQL query that answers the question.
# ### Input:
# Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

# ### Context:
# CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

# ### Response:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


# %% [markdown]
# And the model outputs:
# ```
# SELECT class FROM table_name_12 WHERE frequency_mhz > 91.5 AND city_of_license = "hyannis, nebraska"
# ```
# So it works! If you want to convert your this adapter to a Llama.cpp model to run locally follow my other [guide](https://ragntune.com/blog/A-guide-to-running-Llama-2-qlora-loras-on-Llama.cpp). If you have any questions, shoot me a message on [Elon Musk's website](https://twitter.com/samlhuillier_).
# 


