import torch.nn as nn
import torch.optim as optim
import torch

from model import build_transformer
from dataclasses import dataclass, asdict

from dataset import EnFrDataset, create_dataloader
import time

from tqdm import tqdm

@dataclass
class ModelConfig:
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    src_vocab_size: int = 30000
    trg_vocab_size: int = 30000

@dataclass
class TrainConfig:
    lr: float = 1e-5

@dataclass
class DatasetConfig:
    train_batch_size: int = 16
    val_batch_size: int = 16



if __name__=="__main__":
    m_config = ModelConfig()
    d_config = DatasetConfig()
    t_config = TrainConfig()

    # model
    model = build_transformer(
        **asdict(m_config)
    )

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"Number of parameters: {num_params/1e6}M")
    
    # dataset
    train_dataset = EnFrDataset("train")
    val_dataset = EnFrDataset("validation")
    train_dataloader = create_dataloader(
        train_dataset, batch_size=d_config.train_batch_size
    )
    val_dataloader = create_dataloader(
        val_dataset, batch_size=d_config.val_batch_size
    ) 

    #
    steps = 0
    t0 = time.time()
    device = "cuda"
    mode = torch.compile(model)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), t_config.lr)
    
    epochs = 100
    for epoch in range(epochs):
        # one epoch training
        for batch in tqdm(train_dataloader, desc="Train"):
            x_enc = model.encode(batch["en"].to(device), batch["src_mask"].to(device))
            logits = model.decode(
                batch["fr_in"].to(device),
                x_enc, batch["trg_mask"].to(device),
                batch["src_mask"].to(device)
            )
            steps += 1
            optimizer.zero_grad()
            loss = loss_fn(logits.view((-1, 30000)), batch["fr_out"].to(device).view((-1)))
            loss.backward()
            optimizer.step()
            tqdm.write(f"step: {steps} time: {(time.time()-t0)*1000:.2f}ms loss:{loss.item():.2f}\r")
            t0 = time.time()
            
        # validation
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_dataloader, desc="Validation"):
                x_enc = model.encode(batch["en"].to(device), batch["src_mask"].to(device))
                logits = model.decode(
                    batch["fr_in"].to(device),
                    x_enc, batch["trg_mask"].to(device),
                    batch["src_mask"].to(device)
                )
                loss = loss_fn(logits.view((-1, 30000)), batch["fr_out"].to(device).view((-1)))
                val_loss += (loss.item())/len(val_dataloader)
                tqdm.write(f"Epoch: {epoch} | loss: {val_loss:.2f}")



