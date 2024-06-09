import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import build_tokenizer
import time

from datasets import load_dataset
hf_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")


class EnFrDataset(Dataset):
    def __init__(self, split="train", context_len=512) -> None:
        super().__init__()
        self.split = split
        self.en_tokenizer = build_tokenizer("en")
        self.fr_tokenizer = build_tokenizer("fr")
        self.pairs = hf_dataset[split]["translation"]
        self.context_len = context_len
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        en = pair["en"]
        fr = pair["fr"]
        en_ids = [1,*self.en_tokenizer.encode(en).ids,2] 
        en_ids = torch.tensor((en_ids + [0] * max(0, self.context_len - len(en_ids)))[:self.context_len]).long()
        fr_ids = self.fr_tokenizer.encode(fr).ids
        fr_in = torch.tensor(([1, *fr_ids] + [0] * max(0, self.context_len - len(fr_ids) + 1))[:self.context_len]).long()
        fr_out = torch.tensor(([*fr_ids, 2] + [0] * max(0, self.context_len - len(fr_ids) + 1))[:self.context_len]).long()
        src_mask = (en_ids != 0).float().repeat(self.context_len, 1) 
        trg_mask = (fr_in != 0).float().repeat(self.context_len, 1)
        trg_mask = torch.tril(trg_mask, diagonal=0)
        return {"en":en_ids, "fr_in":fr_in, "fr_out":fr_out, "src_mask":src_mask, "trg_mask": trg_mask}


def collate_fn(data):
    # <pad> = 0, <sos> = 1, and <eos> = 2
    b_size = len(data)
    src, trg_in, trg_out = [], [], []
    src_max_len, trg_max_len = 0, 0

    for i in range(b_size):
        src_seq = [1,*data[i]["en"],2]
        trg = data[i]["fr"]
        trg_in_seq = [1, *trg]
        trg_out_seq = [*trg, 2]
        src += [src_seq]
        trg_in += [trg_in_seq]
        trg_out += [trg_out_seq]
        src_max_len = max(src_max_len, len(src_seq))
        trg_max_len = max(trg_max_len, len(trg_in_seq))

    for i in range(b_size):
        src[i] = src[i] + [0]*(src_max_len-len(src[i]))
        len_trg = trg_max_len-len(trg_in[i])
        trg_in[i] = trg_in[i] + [0]*(len_trg)
        trg_out[i] = trg_out[i] + [0]*(len_trg)

    en = torch.tensor(src, dtype=torch.long)    
    fr_in = torch.tensor(trg_in, dtype=torch.long)
    fr_out = torch.tensor(trg_out, dtype=torch.long)

    # masks
    src_mask = (en != 0).float().repeat(en.size(-1),1,1)\
        .transpose(0,1)
    trg_mask = (fr_in != 0).float()\
        .repeat(fr_in.size(-1),1,1).transpose(0,1)
    trg_mask = torch.tril(trg_mask)
    trg_src_mask = (en != 0).float().repeat(fr_in.size(-1),1,1)\
        .transpose(0,1)

    return {
        "src":en, 
        "trg_in":fr_in,
        "trg_out":fr_out,
        "src_mask":src_mask,
        "trg_mask":trg_mask,
        "trg_src_mask":trg_src_mask
    }

        
def create_dataloader(dataset, batch_size=64, pin_memory=True, num_workers=2):
    shuffle = False
    if dataset.split == "train": shuffle = True
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
          num_workers=num_workers
    )
    return dataloader

        
if __name__=="__main__":
    splits = ["train", "validation", "test"]
    for sp in splits:
        print(f"Creating dataset for {sp}")
        dataset = EnFrDataset(sp)
        print(f"Creating dataloader for {sp}")
        dataloader = create_dataloader(sp)
        start_time = time.time()
        for batch in dataloader:
            src = batch["en"]
        print(f"Processing took {time.time()-start_time:.2f} seconds for split {sp}")
        
            
    

