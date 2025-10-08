import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
import numpy as np
import argparse

# Launch TensorBoard Session
from torch.utils.tensorboard import SummaryWriter


from format_data import prepare_dataloader

def info_nce_loss(hidden, labels, embed_matrix, attention_mask, t=0.2):
    hidden_reshape = hidden.reshape(-1, hidden.shape[-1])
    labels_reshape = labels.reshape(-1)
    mask_reshape = attention_mask.reshape(-1)
    mask_bool = mask_reshape.bool()

    active_hidden = hidden_reshape[mask_bool]
    active_labels = labels_reshape[mask_bool]
    active_logits = active_hidden @ embed_matrix.t()
    active_logits /= t
    loss = F.cross_entropy(active_logits, active_labels, reduction="mean")

    return loss

def kl_loss(logits_student, target, attention_mask, t=2.0):
    view_logits_student = logits_student.reshape(-1, logits_student.shape[-1])
    log_student_probability = F.log_softmax(view_logits_student / t, dim=-1)

    lm_head_layer = model.get_output_embeddings()
    logits_teacher = lm_head_layer(target) # (B, seq_len, vocab_size)
    view_logits_teacher = logits_teacher.reshape(-1, logits_teacher.shape[-1]) # (B * seq_len, vocab_size)
    teacher_probability = F.softmax(view_logits_teacher / t, dim=-1)

    mask_reshaped = attention_mask.reshape(-1) # shape: (B * seq_len)
    mask_bool = mask_reshaped.bool()

    active_log_student_probability = log_student_probability[mask_bool]
    active_teacher_probability = teacher_probability[mask_bool]

    loss = kl_loss_fn(active_log_student_probability, active_teacher_probability)
    scaled_loss = loss * (t * t)
    return scaled_loss

def ce_loss(logits_pred, target, attention_mask):
    lm_head_layer = model.get_output_embeddings()
    logits_target = lm_head_layer(target) # (B, seq_len, vocab_size)
    logits_target = F.softmax(logits_target, dim=-1)
    vocab_size = logits_target.shape[-1]

    pred_reshaped = logits_pred.reshape(-1, vocab_size) # shape: (B * seq_len, vocab_size)
    target_reshaped = logits_target.reshape(-1, vocab_size) # shape: (B * seq_len, vocab_size)
    mask_reshaped = attention_mask.reshape(-1) # shape: (B * seq_len)
    mask_bool = mask_reshaped.bool()

    active_predictions = pred_reshaped[mask_bool]
    active_targets = target_reshaped[mask_bool]
    loss = ce_loss_fn(active_predictions, active_targets)

    return loss


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super().__init__()
        self.custom_embedder = nn.Embedding.from_pretrained(embedding, freeze=True) # frozen

    def forward(self, batch: dict):

        input_ids = batch["input_ids"]
        batch_input_embeds = self.custom_embedder(input_ids)
        return batch_input_embeds

def calculate_loss(input_embeds, input_ids, attention_mask, model):

    # attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)
    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # loss = F.mse_loss(hidden, target)

    # infonce loss
    labels = input_ids[:, 1:]
    hidden = outputs.hidden_states[-1][:, :-1, :]
    hidden = hidden.to(torch.float32)
    attention_mask = attention_mask[:, :-1]
    loss = info_nce_loss(hidden, labels, embed_matrix, attention_mask)

    # celoss
    # logits_pred = outputs.logits[:, :-1, :]
    # target = input_embeds[:, 1:, :]
    # target = target.to(torch.bfloat16)
    # attention_mask = attention_mask[:, :-1]
    # loss = ce_loss(logits_pred, target, attention_mask)

    # # mseloss
    # # hidden = outputs.hidden_states[-1][:, :-1, :] # [batch size, seq len, h dim]
    # # target = input_embeds[:, 1:, :]
    # # attention_mask = attention_mask[:, :-1]
    # # mask_expanded = attention_mask.unsqueeze(-1)
    # # mask_expanded = mask_expanded.expand_as(hidden)
    # # mask_bool = mask_expanded.bool()

    # # active_predictions = hidden[mask_bool]
    # # active_targets = target[mask_bool]
    # # loss = mse_loss(active_predictions, active_targets)

    # # kl
    # # target = input_embeds[:, 1:, :]
    # # target = target.to(torch.bfloat16)
    # # attention_mask = attention_mask[:, : -1]
    # # logits = outputs.logits[:, :-1, :]
    # # loss = kl_loss(logits, target, attention_mask)
    return loss

def prepare_model_and_optimizer(pt_model_path):
    tokenizer = AutoTokenizer.from_pretrained(pt_model_path)
    model = AutoModelForCausalLM.from_pretrained(pt_model_path)

    params_to_optimizer = [param for param in model.parameters() if param.requires_grad]

    llm_params = {
        'lr': 8e-6,
        'params': params_to_optimizer,
        'lr': args.lr,
        'weight_decay': 0.01
    }
    optimizer = torch.optim.AdamW([llm_params])

    return model, tokenizer, optimizer


def evaluate(model, validloader, accelerator: Accelerator, embedder, global_step):
    model.eval()
    all_loss = []
    with torch.no_grad():
        for batch in tqdm(validloader):
            input_embeds = embedder(batch)
            input_embeds = input_embeds.to(torch.bfloat16)
            loss = calculate_loss(input_embeds, batch["input_ids"], batch["attention_mask"], model)
            loss = accelerator.reduce(loss, "mean")
            all_loss.append(loss.item())
            
    test_loss = np.mean(all_loss)
    writer.add_scalar('loss/valid', test_loss, global_step)
    return test_loss

def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, embedder):
    model.train()
    global_step = 1
    epoch = 1

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
        all_loss = []
        for batch in tqdm(trainloader):
            # data input
            input_embeds = embedder(batch)
            optimizer.zero_grad()
            input_embeds = input_embeds.to(torch.bfloat16)
            loss = calculate_loss(input_embeds, batch["input_ids"], batch["attention_mask"], model)
            loss = loss.to(torch.bfloat16)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.reduce(loss, "mean")
            all_loss.append(loss.item())
            writer.add_scalar('loss/train', loss.item(), global_step)
            current_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalar('learning Rate', current_lr, global_step)
            accelerator.print(f"epoch: {e+1}, global_step: {global_step}, lr: {current_lr}, loss: {loss.item()}")
            # accelerator.print(f"epoch: {e+1}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1

            if not global_step % 500:
                # test per epoch
                test_loss = evaluate(model, validloader, accelerator, embedder, global_step)
                accelerator.print(f"epoch: {e+1}, test loss: {test_loss}")
                writer.add_scalar('loss/valid', test_loss, global_step)

        train_loss = np.mean(all_loss)
        accelerator.print(f"epoch: {e+1}, train_loss: {train_loss}")


    writer.close()

    # save model
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    try:
        unwrapped_model.save_pretrained(
            accelerator.project_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True 
        )
    except Exception as e:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(accelerator.project_dir)
        accelerator.print(f"Model has been saved to {output_dir}")

    except Exception as e:
        accelerator.print(f"Error: {e}")


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pt_model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    writer = SummaryWriter('runs/cpt')

    # Prepare dataloader
    data_path = "/train14/psc/permanent/scwang16/datatrain"
    train_loader, valid_loader = prepare_dataloader(data_path, args.batch_size, mode='cpt')
    # test_loader = prepare_dataloader(data_path, B, mode='test')

    # Initialize pretrained models
    pt_model_path = args.pt_model_path

    # Initialize combined model
    model, tokenizer, optimizer = prepare_model_and_optimizer(pt_model_path)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare accelerator
    deepspeed_plugin = DeepspeedPlugin(
        zero_stage=3,
        gradient_accumulation_steps=1,
        offload_optimizer_device="none",
        offload_param_device="none",
    )

    accelerator = Accelerator(project_dir=output_dir,
                            deepspeed_plugin=deepspeed_plugin)
    
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    mse_loss = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()

    device = model.device

    embed_matrix = torch.load("data_process/custom_embedding.pt", weights_only=True).detach().to(device)

    embedder = EmbeddingLayer(embedding=embed_matrix)

    train(model, optimizer, train_loader, valid_loader, accelerator, embedder)