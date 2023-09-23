import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.nn as nn
import torch
import argparse
import os


class Trainset(Dataset):
    def __init__(self, sentence_path, tokenizer):
        super().__init__()
        sents = []
        with open(sentence_path, encoding='utf-8') as f:
            for line in f:
                sent = line.strip()
                if sent:
                    sents.append(sent)

        input_ids, target_ids, mask = [], [], []
        res = tokenizer(sents,truncation=True,max_length=256)
        input_ids = res['input_ids']
        mask = res['attention_mask']
        target_ids = []
        for i, ids in enumerate(input_ids):
            input_ids[i] = ids[:-1]
            target_ids.append(ids[1:])
            mask[i].pop()
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.mask = mask

    def __getitem__(self, i):
        return self.input_ids[i], self.target_ids[i], self.mask[i]

    def __len__(self):
        return len(self.input_ids)


def pad_to_length(sents, length, pad_token_id=0):
    res = []
    for sent in sents:
        sent = sent+[pad_token_id]*(length-len(sent))
        res.append(sent)
    return torch.tensor(res, dtype=torch.int64)


def collate_fn(batch):
    batch_input_ids, batch_target_ids, batch_mask = zip(*batch)
    max_len = max(len(input_ids) for input_ids in batch_input_ids)
    batch_input_ids = pad_to_length(batch_input_ids, max_len, 0)
    batch_target_ids = pad_to_length(batch_target_ids, max_len, 0)
    batch_mask = pad_to_length(batch_mask, max_len, 0)
    return batch_input_ids, batch_target_ids, batch_mask


def train_epoch(epoch, model, dataloader, creterion, optimizer, print_interval):
    local_rank = int(os.environ['LOCAL_RANK'])
    master=(local_rank==0)
    if master:
        print(f'Epoch {epoch}:')
        # 100个batch输出一次
        n_batch = len(dataloader)
        stt0 = time.time()
        cur_batch_loss = 0
        stt1 = time.time()
    
    sampler.set_epoch(epoch)
    device = model.device

    for batch_idx, (input_ids, target_ids, mask) in enumerate(dataloader, 1):
        input_ids = input_ids.to(device)
        target_ids = input_ids.to(device)
        mask = mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        loss = creterion(logits.transpose(1, 2), target_ids)
        loss = loss*mask
        loss = loss.sum()/loss.shape[0]
        # print(loss)
        if master:
            cur_batch_loss += loss.cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if master and batch_idx % print_interval == 0:
            print(f'\t{batch_idx}th batch - {batch_idx}/{n_batch} ({(batch_idx)/n_batch*100:.1f}%)')
            print(f'\t\tcurrent batch loss: {cur_batch_loss:.2f}, cost {time.time()-stt1:.1f} seconds.')
            cur_batch_loss = 0
            stt1=time.time()
    if local_rank==0:
        print(f'Epoch {epoch} cost {time.time()-stt0:.1f} seconds.\n')


if __name__ == "__main__":
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    master=(local_rank==0)
    gpu_id = local_rank
    torch.cuda.set_device(gpu_id)
    n_gpus = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_path', default='./data/sentences.txt')
    parser.add_argument('--model_save_path', default='./model')
    parser.add_argument('--n_epoch', default=4, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--print_interval', default=100, type=int)
    args = parser.parse_args()

    sentence_path = args.sentence_path
    model_save_path = args.model_save_path
    n_epoch = args.n_epoch
    lr = args.lr
    batch_size = args.batch_size
    print_interval = args.print_interval

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # model_name="bigscience/bloom-3b"
    model_name = '/vepfs/lvqs/local_transformers/bloom-3b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).bfloat16()
    model.train()
    model.cuda()
    for name,param in model.named_parameters():
        param.requires_grad=True
    ddp_model = DDP(model)

    creterion = nn.CrossEntropyLoss(reduction='none',label_smoothing=0.1)
    optimizer = AdamW(ddp_model.parameters(), lr=lr)

    print(f'local_rank:{local_rank} begins to build dataset.')
    trainset = Trainset(sentence_path, tokenizer)
    print(f'local_rank:{local_rank} ends to build dataset.')
    dist.barrier()

    sampler = DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size, collate_fn=collate_fn, num_workers=4, sampler=sampler)

    if master:
        print('\nBegin to train model.\n')
        stt=time.time()
    
    for epoch in range(1, n_epoch+1):
        train_epoch(epoch, model=ddp_model, dataloader=trainloader,
                    creterion=creterion, optimizer=optimizer, print_interval=print_interval)
        if master:
            _save_path = os.path.join(model_save_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), _save_path)

    if master:
        print('End to train model.')
        print(f'Totally cost {time.time()-stt:.1f} seconds.\n')

    dist.destroy_process_group()
