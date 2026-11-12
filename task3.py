
import argparse,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
from config import OUTPUT,REFERENCE,CHECKPOINTS
from common import seed_all,tokenizer,load_quant_model,save_json
from data_loading import load_training_examples
from task2 import evaluate
from results import summarize

EPOCHS=3
LEARNING_RATE=2e-4
BATCH_SIZE=1
ACCUMULATE=8

def encode_examples(tok,training):
    eos=tok.convert_tokens_to_ids("<|im_end|>")
    encoded=[]
    for sample in training:
        prefix=tok.apply_chat_template(sample["messages"],tools=sample["tools"],
          tokenize=True,add_generation_prompt=True,enable_thinking=False)
        completion=tok.encode(sample["completion"],add_special_tokens=False)+[eos]
        encoded.append({"ids":prefix+completion,"labels":[-100]*len(prefix)+completion})
    return encoded

def collator(tok):
    def collate(samples):
        length=max(len(s["ids"]) for s in samples)
        pad=tok.pad_token_id or tok.eos_token_id
        return {"input_ids":torch.tensor([s["ids"]+[pad]*(length-len(s["ids"])) for s in samples]),
                "attention_mask":torch.tensor([[1]*len(s["ids"])+[0]*(length-len(s["ids"])) for s in samples]),
                "labels":torch.tensor([s["labels"]+[-100]*(length-len(s["labels"])) for s in samples])}
    return collate

def train_adapter(model,tok,output_dir):
    training=load_training_examples()
    encoded=encode_examples(tok,training)
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
      bias="none",task_type="CAUSAL_LM",
      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()
    loader=DataLoader(encoded,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collator(tok))
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LEARNING_RATE)
    history=[];model.train();t0=time.perf_counter()
    for epoch in range(EPOCHS):
        optimizer.zero_grad();losses=[]
        for step,batch in enumerate(loader,1):
            batch={k:v.to(model.device) for k,v in batch.items()}
            loss=model(**batch).loss
            (loss/ACCUMULATE).backward()
            losses.append(float(loss.detach()))
            if step%ACCUMULATE==0 or step==len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                optimizer.step();optimizer.zero_grad()
        value=sum(losses)/len(losses)
        history.append({"epoch":epoch+1,"mean_loss":value})
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={value:.5f}",flush=True)
    seconds=time.perf_counter()-t0
    path=output_dir/"lora_adapter"
    model.save_pretrained(path)
    save_json(output_dir/"training.json",{"examples":len(training),"epochs":EPOCHS,
      "learning_rate":LEARNING_RATE,"rank":16,"alpha":32,
      "effective_batch_size":BATCH_SIZE*ACCUMULATE,"GPU_training_seconds":seconds,"loss":history})
    model.config.use_cache=True
    model.eval()
    return model

def run(mode,output_dir,adapter_path):
    seed_all();tok=tokenizer();model=load_quant_model()
    output_dir.mkdir(parents=True,exist_ok=True)
    if mode=="train":
        model=train_adapter(model,tok,output_dir)
    else:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter missing: {adapter_path}. Download the optional full project or run --mode train")
        model=PeftModel.from_pretrained(model,str(adapter_path)).eval()
        print("Loaded existing adapter:",adapter_path)
    trained=evaluate(model,tok,"qlora")
    pd.DataFrame(trained).to_csv(output_dir/"task3_qlora_results.csv",index=False)
    
    current_base=OUTPUT/"task2"/"task2_base_results.csv"
    base_path=current_base if current_base.exists() else REFERENCE/"task2_base_results.csv"
    base=pd.read_csv(base_path).to_dict("records")
    print("Comparing against base results from:",base_path)
    summarize(base+trained,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["saved","train"],default="saved")
    parser.add_argument("--adapter-path",type=Path,default=CHECKPOINTS/"lora_adapter")
    parser.add_argument("--output",type=Path,default=OUTPUT/"task3")
    args=parser.parse_args();run(args.mode,args.output,args.adapter_path)
