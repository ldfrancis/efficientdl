

from model import build_transformer
from dataclasses import dataclass, asdict

from dataset import EnFrDataset, create_dataloader
import time

@dataclass
class ModelConfig:
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    src_vocab_size: int = 30000
    trg_vocab_size: int = 30000

@dataclass
class DatasetConfig:
    train_batch_size: int = 1
    val_batch_size: int = 1



if __name__=="__main__":
    m_config = ModelConfig()
    d_config = DatasetConfig()

    # model
    model = build_transformer(
        **asdict(m_config)
    )
    
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
    for batch in train_dataloader:
        # x_enc = model.encode(batch["en"], batch["src_mask"])
        steps += 1
        print(f"step: {steps} time: {time.time()-t0}s")
        t0 = time.time()



