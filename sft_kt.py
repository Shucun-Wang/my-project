import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, mean_squared_error
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm
import numpy as np
import argparse     

# Launch TensorBoard Session
from torch.utils.tensorboard import SummaryWriter

from format_data import import load_dataset, prepare_dataloader


class CombinedModel(nn.Module):
    def __init__(self, pt_model, hidden_size=3584): # 3584 3b 3072 qwen7b:3584 qwen1.5b 1536 0.5b 896
        super().__init__()
        self.pt_model = pt_model
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias) # sigmoid 0.5

    def forward(self, input_embeds, attention_mask, target_embeds):
        
        outputs = self.pt_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # take the last embedding of the sentence to calculate loss
        # hidden = outputs.hidden_states[-1][:, -1, :]
        
        last_hidden_states = outputs.hidden_states[-1]
        # accelerator.print(f"hidden_size is {last_hidden_states.shape}")

        # get true length of the sequence
        hidden_lst = []
        batch_size = attention_mask.shape[0]
        # mask = attention_mask.to(torch.long)
        true_len = torch.sum(mask, dim=-1)

        for i in range(batch_size):
            seq_len = true_len[i].item()
            try:
                one_hidden_state = last_hidden_states[i, seq_len-1, :]
            except IndexError:
                print(f"attention_mask_size:\n{attention_mask.shape}")
                print(f"attention_mask:\n{attention_mask}")
                print(f"input_size:\n{input_embeds.shape}")
                print(true_len)
                print(seq_len)
            # accelerator.print(f"hidden_state_size:\n{one_hidden_state.shape}")
            one_hidden_state = one_hidden_state.unsqueeze(0)
            # accelerator.print(f"hidden_state:\n{one_hidden_state}")
            hidden_lst.append(one_hidden_state)
        hidden = torch.cat(hidden_lst, dim=0)
        # accelerator.print(f"hidden_state:\n{hidden.shape}")

        # preds: [0, 1]
        preds = self.classifier(hidden).to(torch.float32)
        preds = preds.squeeze(1)

        # target
        matrix = embed_matrix[:2, :]
        target_embeds = target_embeds.to(torch.float32)
        targets = torch.argmax(target_embeds @ matrix.t(), dim=1).to(torch.float32)

        return preds, targets


class MultiVectorEmbeddingLayer(nn.Module):
    def __init__(self,
                pt_model,
                embedding: torch.Tensor,
                embedding_dim: int,
                history_placeholder_id: int = -200,
                predict_placeholder_id: int = -300,
                padding_idx: int = 0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.history_placeholder_id = history_placeholder_id
        self.predict_placeholder_id = predict_placeholder_id

        # 1. 
        self.token_embedder = pt_model.get_input_embeddings()
        # 2. 
        self.custom_embedder = nn.Embedding.from_pretrained(embedding, freeze=True) # frozen

    def forward(self, batch: dict, input_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):


        mask = batch["attention_mask"]
        seq = batch["seq"]
        batch_size = mask.shape[0]
        seq_embeds = self.custom_embedder(seq)
        true_len = torch.sum(mask, dim=1)

        output_embedding_list = []
        output_mask_list = []
        output_answer_list = []
        for i in range(batch_size):
            seq_len = true_len[i].item()
            real_seq = seq_embeds[i, :seq_len, :]

            history = real_seq[:-1, :]
            predict = real_seq[-1, :]
            answer = real_seq[:-1, :] 
            output_answer_list.append(answer)

        current_embedding_parts = []
        
        for token_id in input_ids:

            token_int = token_id.item() 
            # print(token_id)
            if token_int >= 0:
                token_tensor = torch.tensor(token_id)
                token_emb = self.token_embedder(token_id).unsqueeze(0)
                
                # print(f"{token_shape}:{token_emb.shape}")
                current_embedding_parts.append(token_emb)
            elif token_int == self.history_placeholder_id:
                # print(f"history: {history.shape}")
                current_embedding_parts.append(history)
            elif token_int == self.predict_placeholder_id:
                # print(f"predict: {predict.shape}")
                current_embedding_parts.append(predict)

            # print(input_ids)
            # print(len(current_embedding_parts))
            final_single_embedding = torch.cat(current_embedding_parts, dim=0)
            # print(final_single_embedding)
            # print(final_single_embedding.shape)
            output_embedding_list.append(final_single_embedding)

            output_mask_list.append(torch.ones(final_single_embedding.shape[0], device=device))

        final_embeddings = pad_sequence(
            output_embedding_list, batch_first=True, padding_value=0.0
        )

        final_attention_mask = pad_sequence(
            output_mask_list, batch_first=True, padding_value=0
        ).long()
        
        final_answer_embeddings = torch.cat(output_answer_list, dim=0)
        return final_embeddings, final_attention_mask, final_answer_embeddings

special_token_mapping = {
    "<history_list>": -200,
    "<predicted_records>": -300,
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
    pt_model = AutoModelForCausalLM.from_pretrained(pt_model_path)
    model = CombinedModel(pt_model=pt_model)
    params_to_optimizer = [param for param in model.pt_model.parameters() if param.requires_grad]
    llm_params = {
        'params': params_to_optimizer,
        'lr': args.lr,
        'weight_decay': 0.01
    }

    linear_params = {
        'params': model.classifier.parameters(),
        'lr': 1e-4,
        'weight_decay': 0.01
    }
    optimizer = torch.optim.AdamW([llm_params, linear_params])

    return model, tokenizer, optimizer


def evaluate(model, validloader, accelerator: Accelerator, embedder, tokens):
    model.eval()
    all_loss = []
    with torch.no_grad():
        for batch in tqdm(validloader):

            input_embeds, attention_mask, target_embeds = embedder(batch, tokens)
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            input_embeds = input_embeds.to(torch.bfloat16)
            attention_mask = attention_mask.to(torch.bfloat16)
            target_embeds = target_embeds.to(torch.bfloat16)
            preds, targets = model(input_embeds, attention_mask, target_embeds)
            loss = loss_fn(preds, targets)
            loss = accelerator.reduce(loss, "mean")
            all_loss.append(loss.item())

    test_loss = np.mean(all_loss)
    return test_loss


def test(model, testloader, accelerator: Accelerator, embedder, tokens):
    model.eval()
    all_target_tensors = []
    all_pred_tensors = []
    with torch.no_grad():
        for batch in tqdm(testloader):
            input_embeds, attention_mask, target_embeds = embedder(batch, tokens)
            input_embeds = input_embeds.to(torch.bfloat16)
            attention_mask = attention_mask.to(torch.bfloat16)
            target_embeds = target_embeds.to(torch.bfloat16)
            preds, targets = model(input_embeds, attention_mask, target_embeds)

            # hidden and target_embeds are torch.float32 type
            all_target_tensors.append(targets)
            all_pred_tensors.append(preds)

    all_target = torch.cat(all_target_tensors, dim=0)
    all_pred = torch.cat(all_pred_tensors, dim=0)

    gathered_targets = accelerator.gather_for_metrics(all_target)
    gathered_preds = accelerator.gather_for_metrics(all_pred)

    if accelerator.is_main_process:
        preds_np = gathered_preds.cpu().numpy()
        targets_np = gathered_targets.cpu().numpy()
        auc = roc_auc_score(targets_np, preds_np)
        print('auc: ' + str(auc))

        mse = mean_squared_error(targets_np, preds_np)
        rmse = np.sqrt(mse)
        print('rmse: ' + str(rmse))

        preds_np_int = (preds_np > 0.5).astype(int)
        targets_np_int = targets_np.astype(int)
        acc = accuracy_score(targets_np_int, preds_np_int)
        print('acc: ' + str(acc))
        
        recall = recall_score(targets_np_int, preds_np_int)
        precision = precision_score(targets_np_int, preds_np_int)
        f1 = f1_score(targets_np_int, preds_np_int)
        print('precision: ' + str(precision))
        print('recall: ' + str(recall))
        print('f1: ' + str(f1))


def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, embedder, tokens):
    model.train()
    global_step = 1
    epoch = 2

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
            input_embeds, attention_mask, target_embeds = embedder(batch, tokens)
            optimizer.zero_grad()
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            input_embeds = input_embeds.to(torch.bfloat16)
            attention_mask = attention_mask.to(torch.bfloat16)

    for e in range(epoch):
        model.train()
        for batch in tqdm(trainloader):
            # Prepare input
            input_embeds, attention_mask, target_embeds = embedder(batch, tokens)
            optimizer.zero_grad()
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            input_embeds = input_embeds.to(torch.bfloat16)
            attention_mask = attention_mask.to(torch.bfloat16)
            target_embeds = target_embeds.to(torch.bfloat16)
            preds, targets = model(input_embeds, attention_mask, target_embeds)
            loss = loss_fn(preds, targets)
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

            if not global_step % 50:
                accelerator.print(f"----------valid set----------")
                valid_loss = evaluate(model, valid_loader, accelerator, embedder, tokens)
                accelerator.print(f"epoch: {e+1}, global_step: {global_step}, loss: {valid_loss}")
                writer.add_scalar('loss/valid', valid_loss, global_step)
                # test per epoch
                accelerator.print(f"----------test set----------")
                test(model, test_loader, accelerator, embedder, tokens)
        
        # test_loss = evaluate(model, validloader, accelerator, embedder, tokens)
        # accelerator.print(f"epoch: {e+1}, test_loss: {test_loss}")


    writer.close()

    # save model
    accelerator.wait_for_everyone()

    full_state_dict = accelerator.get_state_dict(model) # 
    # print(f"Get full state dict over.")
    # print(type(full_state_dict))
    # print(full_state_dict)
    if accelerator.is_main_process:
        qwen_state_dict = {
            key.replace("pt_model.", ""): value
            for key, value in full_state_dict.items()
            if key.startswith("pt_model.")
        }
        print(f"fix qwen state dict over.")
        
        classifier_state_dict = {
            key.replace("classifier.", ""): value
            for key, value in full_state_dict.items()
            if key.startswith("classifier.")
        }
        print(f"fix classifier state dict over.")

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

        classifier_path = f"{output_dir}/classifier.pt"
        torch.save(classifier_state_dict, classifier_path)

        tokenizer.save_pretrained(qwen_model_path)


if __name__ == "__main__":
    # -- Hyperparameters --
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--pt_model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    
    writer = SummaryWriter('runs/sft')

    # Prepare dataloader
    data_path = ""
    train_loader, valid_loader = prepare_dataloader(data_path, args.batch_size, mode="sft")
    test_loader = prepare_dataloader(data_path, args.batch_size, mode="test")

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

    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader, test_loader)
    loss_fn = nn.BCELoss()
    # Prepare inputs
    device = model.pt_model.device

    embed_matrix = torch.load("data_process/custom_embedding.pt", weights_only=True).detach().to(device)
    multi_vec_embedder = MultiVectorEmbeddingLayer(
        pt_model=model.pt_model,
        embedding=embed_matrix,
        embedding_dim=3584,
    )


    # sft data
    input_string = "Given a student historical learning interactions :<history_list>, please predict the response according to <predicted_record>"
    tokens = custom_tokenize(input_string, tokenizer).to(device)


    train(model, optimizer, train_loader, valid_loader, accelerator, multi_vec_embedder, tokens)