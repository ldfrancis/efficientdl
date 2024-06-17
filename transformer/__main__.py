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
    lr: float = 3e-4

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
    model.to(device)
    mode = torch.compile(model)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), t_config.lr)
    
    epochs = 100
    for epoch in range(epochs):
        train_loss = 0
        step = 0
        n_steps = len(train_dataloader)
        # one epoch training
        with tqdm(total=len(train_dataloader)) as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()
                x_enc = model.encode(batch["en"].to(device), batch["src_mask"].to(device))
                logits = model.decode(
                    batch["fr_in"].to(device),
                    x_enc, batch["trg_mask"].to(device),
                    batch["src_mask"].to(device)
                )
                loss = loss_fn(logits.view((-1, 30000)), batch["fr_out"].to(device).view((-1)))
                loss.backward()
                optimizer.step()
                step += 1
                train_loss += (loss.item()-train_loss)/step
                pbar.set_postfix(
                    {
                        "time": f"{(time.time()-t0)*1000:.2f}ms",
                        "loss": f"{train_loss:.2f}",
                    }
                )
                pbar.update(1)
                t0 = time.time()
            
        # validation
        with torch.no_grad():
            val_loss = 0
            step = 0
            with tqdm(total=len(val_dataloader), desc="Validation") as pbar:
                for batch in val_dataloader:
                    x_enc = model.encode(batch["en"].to(device), batch["src_mask"].to(device))
                    logits = model.decode(
                        batch["fr_in"].to(device),
                        x_enc, batch["trg_mask"].to(device),
                        batch["src_mask"].to(device)
                    )
                    loss = loss_fn(logits.view((-1, 30000)), batch["fr_out"].to(device).view((-1)))
                    step += 1
                    val_loss += (loss.item()-val_loss)/step
                    pbar.set_postfix(
                        {"Epoch": epoch, "loss": f"{val_loss:.2f}"}
                    )
                    pbar.update(1)



