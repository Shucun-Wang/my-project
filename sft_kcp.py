import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataloader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, get_scheduler
from tqdm import tqdm
import numpy as np
import argparse

# Launch TensorBoard Session
from torch.utils.tensorboard import SummaryWriter


from q_former import SentenceQFormer


class CombinedModel(nn.Module):
    def __init__(self, pt_model, q_former, knowledge_matrix, hidden_size=3584): # 2048 3b 3072 qwen7b:3584 qwen1.5b 1536 0.5b 896
        super().__init__()
        self.pt_model = pt_model
        self.hidden_size = hidden_size
        self.q_former = q_former
        self.knowledge_matrix = nn.Parameter(knowledge_matrix, requires_grad=True)
        
        self.token_embedder = pt_model.get_input_embeddings()
        self.exercise_placeholder_id: int = -200

    def forward(self, batch, tokens):
        mask = batch["attention_mask"]
        exercise_id = batch["input_ids"]
        batch_size = mask.shape[0]
        device = exercise_id.device
        embeds_ = self.token_embedder(exercise_id)
        e_embeds = self.q_former(embeds_, mask)
        e_embeds = e_embeds.squeeze(1)
        
        output_embedding_list = []
        output_mask_list = []
        for i in range(batch_size):
            current_embedding_parts = []
            
            for token_id in tokens:

                token_int = token_id.item() 
                if token_int >= 0:
                    token_emb = self.token_embedder(token_id).unsqueeze(0)
                    current_embedding_parts.append(token_emb)
                elif token_int == self.exercise_placeholder_id:
                    current_embedding_parts.append(e_embeds[i].unsqueeze(0))

            final_single_embedding = torch.cat(current_embedding_parts, dim=0)
            output_embedding_list.append(final_single_embedding)

            output_mask_list.append(torch.ones(final_single_embedding.shape[0], device=device))

        final_embeddings = pad_sequence(
            output_embedding_list, batch_first=True, padding_value=0.0
        )
        
        final_attention_mask = pad_sequence(
            output_mask_list, batch_first=True, padding_value=0
        ).long()

        outputs = self.pt_model(
            inputs_embeds=final_embeddings,
            attention_mask=final_attention_mask,
            output_hidden_states=True,
        )
        question_embeddings = outputs.last_hidden_state[:, -1, :]

        q_norm = F.normalize(question_embeddings, p=2, dim=1)
        k_norm = F.normalize(self.knowledge_matrix, p=2, dim=1)

        # [batch_size, 768] @ [768, 4795] -> [batch_size, 4795]
        logits = q_norm @ k_norm.T

        return logits / args.temp

special_token_mapping = {
    "<exercise>": -200,
}


def custom_tokenize(text, tokenizer):
    tokens = []
    special_tokens = list(special_token_mapping.keys())
    all_parts = [text]
    for special_token in special_tokens:
        new_parts = []
        for part in all_parts:
            sub_parts = part.split(special_token)
            for i, sub_part in enumerate(sub_parts):
                if sub_part:
                    new_parts.append(sub_part)
                if i < len(sub_parts) - 1:
                    new_parts.append(special_token)
        all_parts = new_parts[:]


    for part in all_parts:
        if part in special_tokens:
            tokens.append(special_token_mapping[part])
        else:
            tokens.extend(tokenizer(part, add_special_tokens=False).input_ids)
    return torch.tensor(tokens, dtype=torch.long) # input_ids


def prepare_model_and_optimizer(pt_model_path):
    tokenizer = AutoTokenizer.from_pretrained(pt_model_path)
    pt_model = AutoModel.from_pretrained(pt_model_path)

    q_former = SentenceQFormer(
        dim=3584,
        num_queries=1,
        num_layers=10,
        num_heads=8
    )

    adapter_weights = torch.load("", weights_only=True)
    q_former.load_state_dict(adapter_weights)
    knowledge_matrix = torch.load("", weights_only=True).cpu()
    model = CombinedModel(pt_model, q_former, knowledge_matrix)

    llm_params = {
        'params': [param for param in model.pt_model.parameters() if param.requires_grad],
        'lr': args.lr,
        'weight_decay': 0.01
    }

    q_former_params = {
        'params': model.q_former.parameters(),
        'lr': 5e-5,
        'weight_decay': 0.01
    }

    knowledge_matrix_params = {
        'params': [model.knowledge_matrix],
        'lr': 5e-5,
        'weight_decay': 0.01
    }
    
    optimizer = torch.optim.AdamW([llm_params, q_former_params, knowledge_matrix_params])

    return model, tokenizer, optimizer


@torch.no_grad()
def evaluate(model, validloader, accelerator: Accelerator, tokens):
    model.eval()
    all_loss = []
    
    with torch.no_grad():
        for batch, labels in tqdm(validloader):
            logits = model(batch, tokens)
            loss = F.cross_entropy(logits, labels)
            loss = accelerator.reduce(loss, "mean")
            all_loss.append(loss.item())
        
        test_loss = np.mean(all_loss)
    return test_loss


@torch.no_grad()
def test(model, testloader, accelerator: Accelerator, tokens):
    model.eval()

    all_preds, all_golds, all_top5_preds = [], [], []
    for batch, labels in tqdm(testloader):
        logits = model(batch, tokens)

        # Top-1 
        preds = logits.argmax(dim=1)
        all_preds.append(preds)

        # Top-5 
        _, top5 = torch.topk(logits, 5, dim=1)
        all_top5_preds.append(top5)
        
        # labels
        all_golds.append(labels)


    preds = torch.cat(all_preds)
    golds = torch.cat(all_golds)
    top5_preds = torch.cat(all_top5_preds)

    gathered_targets = accelerator.gather_for_metrics(golds)
    gathered_preds = accelerator.gather_for_metrics(preds)
    gathered_top5_preds = accelerator.gather_for_metrics(top5_preds)

    if accelerator.is_main_process:
        # Top-1 acc
        acc = (gathered_preds == gathered_targets).float().mean().item()
        
        # Top-5 acc
        acc5 = (gathered_targets.unsqueeze(1) == gathered_top5_preds).any(dim=1).float().mean().item()
        
        macro_f1 = f1_score(gathered_targets.cpu().numpy(), gathered_preds.cpu().numpy(), average='macro') 
        weighted_f1 = f1_score(gathered_targets.cpu().numpy(), gathered_preds.cpu().numpy(), average='weighted') 
        print(f"acc: {acc}, acc5: {acc5}, macro_f1: {macro_f1}, weighted_f1: {weighted_f1}")

    return acc, acc5, macro_f1, weighted_f1


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, tokens):
    model.train()
    global_step = 1
    epoch = 8

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
        for batch, labels in tqdm(trainloader):
            # Prepare input
            optimizer.zero_grad()
            logits = model(batch, tokens)
            loss = F.cross_entropy(logits, labels)
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

        # validation
        accelerator.print(f"----------valid set----------")
        valid_loss = evaluate(model, valid_loader, accelerator, tokens)
        accelerator.print(f"epoch: {e+1}, global_step: {global_step}, loss: {valid_loss}")
        writer.add_scalar('loss/valid', valid_loss, global_step)
        # test
        accelerator.print(f"----------test set----------")
        test(model, test_loader, accelerator, tokens)

    # test_loss = evaluate(model, validloader, accelerator, embedder, tokens)
    # accelerator.print(f"epoch: {e+1}, test_loss: {test_loss}")

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

        # classifier_state_dict = {
        #     key.replace("classifier.", ""): value
        #     for key, value in full_state_dict.items()
        #     if key.startswith("classifier.")
        # }
        # print(f"fix classifier state dict over.")

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

        # classifier_path = f"{output_dir}/classifier.pt"
        # torch.save(classifier_state_dict, classifier_path)

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
    
    writer = SummaryWriter('runs/sft_ap')
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
                            
    # Initialize pretrained models
    pt_model_path = args.pt_model_path

    # Initialize CombinedModel
    model, tokenizer, optimizer = prepare_model_and_optimizer(pt_model_path)
    
    # Prepare dataloader
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []
    test_labels = []

    with open("", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            train_texts.append(line["text"])
            train_labels.append(line["label"])

    with open("", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            val_texts.append(line["text"])
            val_labels.append(line["label"])


    with open("", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            test_texts.append(line["text"])
            test_labels.append(line["label"])


    class QDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding='max_length',
                                max_length=50, return_tensors='pt')
            
            self.labels = torch.tensor(labels)

        def __len__(self): return len(self.labels)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


    train_loader = Dataloader(QDataset(train_texts, train_labels),
                            batch_size=args.batch_size, shuffle=True)
                            
    valid_loader = Dataloader(QDataset(val_texts, val_labels),
                            batch_size=args.batch_size)

    test_loader = Dataloader(QDataset(test_texts, test_labels),
                            batch_size=args.batch_size)


    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader, test_loader)

    # Prepare inputs
    device = model.pt_model.device

    # sft data
    input_string = "Given the text of an exercise :<exercise>, please give me the related knowledge concept to this exercise"
    tokens = custom_tokenize(input_string, tokenizer).to(device)


    train(model, optimizer, train_loader, valid_loader, accelerator, tokens)