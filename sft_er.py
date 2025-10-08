import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataloader, Dataset
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, get_scheduler
from tqdm import tqdm
import numpy as np
import argparse

# Launch TensorBoard Session
from torch.utils.tensorboard import SummaryWriter


class SFTClassificationDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        text = item['input']
        label = item['output']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }


class CombinedModel(nn.Module):
    def __init__(self, pt_model, exercise_representation, hidden_size=3584): # 2048 3b 3072 qwen7b:3584 qwen1.5b 1536 0.5b 896
        super().__init__()
        self.pt_model = pt_model
        self.hidden_size = hidden_size
        self.exercise_representation = nn.Embedding.from_pretrained(exercise_representation)
        
        self.token_embedder = pt_model.get_input_embeddings()
        
    def forward(self, batch):
        mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        label_embeds = self.exercise_representation(labels)
        batch_size = mask.shape[0]
        device = input_ids.device
        
        outputs = self.pt_model(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
        
        last_hidden_states = outputs.last_hidden_state
        
        # get true length of the sequence
        hidden_lst = []
        batch_size = mask.shape[0]
        mask_index = mask.to(torch.long)
        true_len = torch.sum(mask_index, dim=1)
        for i in range(batch_size):
            seq_len = true_len[i].item()
            one_hidden_state = last_hidden_states[i, seq_len-1, :]
            one_hidden_state = one_hidden_state.unsqueeze(0)
            hidden_lst.append(one_hidden_state)
        last_embedding = torch.cat(hidden_lst, dim=0)
        
        # last_embedding = outputs.last_hidden_state[:, -1, :]
        loss = mse_loss_fn(label_embeds, last_embedding)
        
        return loss

        
def prepare_model_and_optimizer(pt_model_path):
    tokenizer = AutoTokenizer.from_pretrained(pt_model_path)
    pt_model = AutoModel.from_pretrained(pt_model_path)
    embed_matrix = torch.load("", weights_only=True).detach()
    exercise_representation = embed_matrix[:4036] + [30095]
    model = CombinedModel(pt_model, exercise_representation)

    llm_params = {
        'params': [param for param in model.pt_model.parameters() if param.requires_grad],
        'lr': args.lr,
        'weight_decay': 0.01
    }

    exercise_representation = {
            'params': [param for param in model.exercise_representation.parameters() if param.requires_grad],
            'lr': 5e-5,
            'weight_decay': 0.01
        }

    optimizer = torch.optim.AdamW([llm_params, exercise_representation])

    return model, tokenizer, optimizer


def train(model, optimizer, trainloader, accelerator: Accelerator):
    model.train()
    global_step = 1
    epoch = 3

    # get scheduler
    num_training_steps = epoch * len(trainloader)
    warmup_ratio = 0.05
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    lr_scheduler = get_scheduler(
        name="cosine", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    for e in range(epoch):
        model.train()
        for batch in tqdm(trainloader):
            # Prepare input
            optimizer.zero_grad()
            loss = model(batch)
            loss = loss.to(torch.bfloat16)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.reduce(loss, "mean")
            writer.add_scalar('loss/train', loss.item(), global_step)
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar('learning Rate', current_lr, global_step)
            accelerator.print(f"epoch: {e+1}, global_step: {global_step}, lr: {current_lr}, loss: {loss.item()}")
            global_step += 1

    writer.close()

    # save model
    accelerator.wait_for_everyone()

    full_state_dict = accelerator.get_state_dict(model) # 

    if accelerator.is_main_process:
        qwen_state_dict = {
            key.replace("pt_model.", ""): value
            for key, value in full_state_dict.items()
            if key.startswith("pt_model.")
        }
        print(f"fix qwen state dict over.")

        exercise_representation_dict = {
            key.replace("exercise_representation.", ""): value
            for key, value in full_state_dict.items()
            if key.startswith("exercise_representation.")
        }
        print(f"fix exercise_representation state dict over.")

    accelerator.wait_for_everyone()

    unwrapped = accelerator.unwrap_model(model)
    qwen_model_path = f"{output_dir}/qwen_model"

    if accelerator.is_main_process:
        unwrapped.pt_model.save_pretrained(
            qwen_model_path,
            save_function=accelerator.save,
            state_dict=qwen_state_dict,
            safe_serialization=True,
        )

        exercise_representation_path = f"{output_dir}/exercise_representation.pt"
        torch.save(exercise_representation_dict, exercise_representation_path)

        tokenizer.save_pretrained(qwen_model_path)


if __name__ == "__main__":
    # --Hyperparameters--
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--pt_model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    
    writer = SummaryWriter('runs/sft_er')

    # Initialize pretrained models
    pt_model_path = args.pt_model_path
    
    # Initialize CombinedModel
    model, tokenizer, optimizer = prepare_model_and_optimizer(pt_model_path)
    mse_loss_fn = nn.MSELoss()
    
    JSONL_FILE = ""
    train_dataset = SFTClassificationDataset(JSONL_FILE, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = Dataloader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator, 
        shuffle=True 
    )
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare accelerator
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=3,
        gradient_accumulation_steps=1,
        offload_optimizer_device="none",
        offload_param_device="none",
    )


    accelerator = Accelerator(project_dir=output_dir,
                            deepspeed_plugin=deepspeed_plugin)
                            
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


    train(model, optimizer, train_loader, accelerator)