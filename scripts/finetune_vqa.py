#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL-2B for Remote Sensing VQA — SatQuery EvidenceSwarm (SIH26167)
Supports Unsloth (if available) else fallback to PEFT + TRL + Transformers (Windows/CUDA)
"""
import os, json, argparse, random, gc
from pathlib import Path
import torch
from PIL import Image
import numpy as np

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_TRAIN = "data/vqa_train.jsonl"
DEFAULT_VAL = "data/vqa_val.jsonl"
OUTPUT_DIR = "models/qwen2vl_vqa_lora"

def get_lora_config():
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("peft required: pip install peft")
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_model_and_processor(use_4bit=True):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    print(f"[Load] Model={MODEL_ID} 4bit={use_4bit}")
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = dict(trust_remote_code=True)
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    try:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
    except Exception as e:
        print(f"[Load] Qwen2VL failed {e}, trying AutoModelForCausalLM fallback")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("[Load] Processor ready")
    return model, processor

def build_text_prompt(question, answer=None, for_training=True):
    if for_training:
        return f"<|im_start|>user\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    else:
        return f"<|im_start|>user\nQuestion: {question}\n<|im_end|>\n<|im_start|>assistant\n"

def load_jsonl(path, limit=None):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except: continue
            if limit and len(rows)>=limit: break
    return rows

def dummy_image(size=224):
    arr = np.random.randint(0,255,(size,size,3),dtype=np.uint8)
    return Image.fromarray(arr)

def make_dataset(rows, processor, max_seq=2048):
    texts=[]
    for r in rows:
        q=r.get("question","")
        a=r.get("answer","")
        texts.append(build_text_prompt(q,a,True))
    # Tokenize
    enc = processor.tokenizer(
        texts,
        truncation=True,
        max_length=max_seq,
        padding=False,
    )
    # Create dataset as list of dicts
    dataset=[]
    for i, ids in enumerate(enc["input_ids"]):
        dataset.append({"input_ids": ids, "labels": ids.copy(), "attention_mask": enc["attention_mask"][i]})
    return dataset

class SimpleCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id=pad_token_id
    def __call__(self, features):
        max_len=max(len(f["input_ids"]) for f in features)
        input_ids=[]
        labels=[]
        masks=[]
        for f in features:
            pad_len=max_len-len(f["input_ids"])
            input_ids.append(f["input_ids"]+[self.pad_token_id]*pad_len)
            labels.append(f["labels"]+[-100]*pad_len)
            masks.append(f["attention_mask"]+[0]*pad_len)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long)}

def print_vram(tag=""):
    if torch.cuda.is_available():
        alloc=torch.cuda.memory_allocated()/1024**3
        reserved=torch.cuda.memory_reserved()/1024**3
        peak=torch.cuda.max_memory_allocated()/1024**3
        print(f"[VRAM{tag}] alloc {alloc:.2f}GB reserved {reserved:.2f}GB peak {peak:.2f}GB")
        return alloc
    return 0

def try_unsloth_train(args, rows):
    try:
        from unsloth import FastVisionModel
        print("[Unsloth] Available — using FastVisionModel")
        model, processor = FastVisionModel.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        from peft import get_peft_model
        lora_cfg=get_lora_config()
        model=get_peft_model(model, lora_cfg)
        # Minimal SFTTrainer path
        from trl import SFTTrainer
        from transformers import TrainingArguments
        texts=[build_text_prompt(r["question"], r["answer"]) for r in rows]
        ds_raw={"text":texts}
        import datasets
        ds=datasets.Dataset.from_dict(ds_raw)
        training_args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            num_train_epochs=args.epochs if not args.dry_run else 1,
            max_steps=5 if args.dry_run else -1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            optim="adamw_8bit",
            max_seq_length=args.max_seq,
            report_to="none",
        )
        trainer=SFTTrainer(model=model, train_dataset=ds, args=training_args, dataset_text_field="text", max_seq_length=args.max_seq)
        trainer.train()
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"[Unsloth] Saved to {args.output_dir}")
        return True
    except Exception as e:
        print(f"[Unsloth] not usable: {e}")
        import traceback; traceback.print_exc()
        return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--val", default=DEFAULT_VAL)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_seq", type=int, default=2048)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dry_run", action="store_true", help="10 samples, 5 steps")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--no_4bit", action="store_true")
    args=ap.parse_args()

    for p in [args.train, args.val]:
        if not Path(p).exists():
            print(f"[Error] {p} not found"); return

    rows=load_jsonl(args.train, limit=args.max_samples or (10 if args.dry_run else None))
    if args.dry_run:
        rows=rows[:10]
        print(f"[Dry Run] {len(rows)} samples only")

    os.makedirs(args.output_dir, exist_ok=True)
    print_vram(" before load")
    # Try Unsloth first
    if not args.no_4bit:
        ok=try_unsloth_train(args, rows)
        if ok:
            print_vram(" after unsloth")
            return

    # Fallback PEFT + Trainer
    print("[Fallback] PEFT+Transformers Trainer")
    model, processor = load_model_and_processor(use_4bit=not args.no_4bit)
    print_vram(" after load")
    from peft import get_peft_model, prepare_model_for_kbit_training
    try:
        model=prepare_model_for_kbit_training(model)
    except Exception as e:
        print(f"[Warn] prepare_kbit {e}")
    lora_cfg=get_lora_config()
    model=get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print_vram(" after LoRA")

    dataset=make_dataset(rows, processor, max_seq=args.max_seq)
    # Enable grad checkpoint to save VRAM
    try:
        model.gradient_checkpointing_enable()
        print("[Info] gradient_checkpointing enabled")
    except: pass

    # If still OOM risk, auto fallback suggestions already in memory; keep grad_acc as is

    from transformers import TrainingArguments, Trainer
    max_steps = 5 if args.dry_run else -1
    num_epochs = 1 if args.dry_run else args.epochs

    # Adjust save_steps for dry run
    save_steps = 2 if args.dry_run else args.save_steps
    logging_steps = 1

    training_args=TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    collator=SimpleCollator(processor.tokenizer.pad_token_id if hasattr(processor,'tokenizer') else 0)

    # Tiny torch dataset
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data=data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    train_ds=ListDataset(dataset)
    trainer=Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)

    print(f"[Train] start dry_run={args.dry_run} samples={len(rows)} epochs={num_epochs} steps={max_steps}")
    print_vram(" before train")
    try:
        res=trainer.train()
        print(f"[Train] done {res}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] {e}")
        print("Try: --grad_acc 8 --max_seq 1024 or subsample to 3000")
        raise
    print_vram(" after train")
    # Check VRAM budget
    alloc=print_vram(" final")
    if alloc>7.5:
        print(f"[Warn] VRAM {alloc:.2f} >7.5GB budget")
    else:
        print(f"[OK] VRAM {alloc:.2f} within 7.5GB")

    # Save adapter
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    # Also save tokenizer explicitly
    try:
        processor.tokenizer.save_pretrained(args.output_dir)
    except: pass
    print(f"[Save] adapter -> {args.output_dir}")
    # Quick file list
    for p in Path(args.output_dir).glob("*"):
        print(f"  {p.name} {(p.stat().st_size/1024):.1f}KB")
    print("[Done] Loss should be decreasing (see logs above)")

if __name__=="__main__":
    main()
