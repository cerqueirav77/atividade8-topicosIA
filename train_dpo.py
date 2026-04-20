# train_dpo.py
# Laboratório 08 — Alinhamento Humano com DPO
# Autor: Victor Cerqueira

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# ── Configurações gerais ──────────────────────────────────────────────────────
MODEL_NAME   = "pierreguillou/gpt2-small-portuguese"
DATASET_PATH = "data/hhh_dataset.jsonl"
OUTPUT_DIR   = "output/dpo-hhh"
BETA         = 0.1   # Hiperparâmetro que controla a divergência KL

# ── Carregamento do tokenizer ─────────────────────────────────────────────────
print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── Carregamento do modelo ator ───────────────────────────────────────────────
print("Carregando modelo ator...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)
model.config.use_cache = False

# ── Adaptador LoRA sobre o modelo ator ───────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn", "c_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Modelo de referência (congelado) ─────────────────────────────────────────
print("Carregando modelo de referência (congelado)...")
model_ref = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)
model_ref.config.use_cache = False

# ── Dataset de preferências ───────────────────────────────────────────────────
print("Carregando dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ── Configuração do treinamento ───────────────────────────────────────────────
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    beta=BETA,
    max_length=256,
    max_prompt_length=128,
    report_to="none",
)

# ── Treinador DPO ─────────────────────────────────────────────────────────────
print("Inicializando DPOTrainer...")
trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# ── Treinamento ───────────────────────────────────────────────────────────────
print("Iniciando treinamento...")
trainer.train()

# ── Salvamento do modelo alinhado ─────────────────────────────────────────────
print(f"Salvando modelo em {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Treinamento concluído!")