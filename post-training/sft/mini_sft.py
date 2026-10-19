import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model


model_name = "Qwen/Qwen3-1.7B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

device = "cuda"
model.to(device)

dataset_name = "yahma/alpaca-cleaned"
ds = load_dataset(dataset_name, split="train")
ds_train = ds.select(range(10000))
ds_eval = ds.select(range(10000, 10100))

def build_prompt(instruction, input_text=""):
    if input_text:
        user_text = (f"{instruction}\n\nInput:\n{input_text}")
    else:
        user_text = instruction

    return [{"role": "user", "content": user_text}]

def tokenize_item(item, max_len = 1024):
    prompt_messages = build_prompt(item["instruction"], item["input"])

    full_messages = prompt_messages + [{"role": "assistant", "content": item["output"]}]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict = False
    )

    input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict = False
    )

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return { "input_ids": input_ids, "labels": labels }

td = ds_train.map(tokenize_item, remove_columns = ds_train.column_names)
ed = ds_eval.map(tokenize_item, remove_columns = ds_eval.column_names)

def collate_fn(batch):
    result = tokenizer.pad(
        [{"input_ids": x["input_ids"]} for x in batch],
        padding=True,
        return_tensors="pt",
    )

    result["labels"] = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=-100,
    )

    return result

tl = DataLoader(td, batch_size=1, shuffle=True, collate_fn=collate_fn)
el = DataLoader(ed, batch_size=1, shuffle=False, collate_fn=collate_fn)


def generate(instruction, input_text="", max_new_tokens=80):
    model.eval()

    messages = build_prompt(instruction, input_text)

    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0][ids["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(text)

optimizer = AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4,
)

gradient_accumulation_steps = 8

def training(epoches = 1):
    model.train()
    for epoch in range(epoches):
        optimizer.zero_grad()
        for step, batch in enumerate(tl):
            batch = { k: v.to(device) for k, v in batch.items() }
            output = model(**batch)
            loss = output.loss / gradient_accumulation_steps
            loss.backward()

            is_last_step = step == len(tl) - 1
            if (step + 1) % gradient_accumulation_steps == 0 or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"training loss = {loss.item() * gradient_accumulation_steps:.4f}")

def evaluation():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(el):
             batch = { k: v.to(device) for k, v in batch.items() }
             output = model(**batch)
             loss = output.loss
             total_loss += loss.item()
        print(f"eval loss = {total_loss / len(el):.4f}")

evaluation()
generate("The capital of France?")
training()
evaluation()
generate("The capital of France?")

model.save_pretrained("checkpoints/qwen3-1.7b-base-lora")
tokenizer.save_pretrained("checkpoints/qwen3-1.7b-base-lora")
