from dataset import create_dataloader
from torch.nn.functional import log_softmax
from tqdm import tqdm
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(transformer, data):
    device = get_device()
    x_src = data["en"].to(device)
    x_trg = data["fr_in"].to(device)
    src_mask = data["src_mask"].to(device)
    trg_mask = data["trg_mask"].to(device)
    x_enc = transformer.encode(x_src, src_mask)
    logits = transformer.decode(x_trg, x_enc, trg_mask, src_mask)
    return logits


def translate(transformer, batch, src_tokenizer, trg_tokenizer, max_len=512):
    # encode
    device = get_device()
    x_src = batch["en"].to(device)
    b_size = x_src.shape[0]
    src_mask = batch["src_mask"].to(device)
    x_enc = transformer.encode(x_src, src_mask)
    x_trg = torch.ones((b_size, 1)).long().to(device)
    trg_mask = torch.ones((b_size, 1,1)).float().to(device)
    dones = torch.zeros((b_size,))
    printed = torch.zeros((b_size,))
    for i in range(max_len):
        logits = transformer.decode(x_trg, x_enc, trg_mask, src_mask[:,:i+1,:])
        ids = logits[:,-1,:].argmax(dim=-1).squeeze().long()
        dones[(ids == 2).argwhere()] = 1
        ids[dones.argwhere()] = 0
        x_trg = torch.cat([x_trg, ids.unsqueeze(1)], dim=-1)
        trg_mask = torch.ones((b_size, x_trg.size(1), x_trg.size(1))).float()
        
        
        if torch.sum(dones).item() == b_size:
           break

    for j in [0,1,2]:
        if printed[j].item() == 0:
            en = src_tokenizer.decode(x_src[j].cpu().numpy())
            tqdm.write(
                f"Source: {en}\n"
                f"Predicted:{trg_tokenizer.decode(x_trg[j].cpu().numpy())}\n"
                f"Target:{trg_tokenizer.decode(batch['fr_in'][j].cpu().numpy())}\n"
            )
            printed[j] = 1

def validate(transformer, val_dataloader, loss_fn, src_tokenizer, trg_tokenizer):
    val_loss = 0
    progress_bar = tqdm(total=len(val_dataloader), ncols=100, desc="Validating")
    transformer.eval()
    device = get_device()
    with torch.no_grad():
        for i,batch in enumerate(val_dataloader):
            logits = infer(transformer, batch)
            log_likelihood = log_softmax(logits, dim=-1)
            loss = loss_fn(log_likelihood.transpose(1,2), batch["fr_out"].to(device))
            val_loss = val_loss + (1/(i+1))*(loss.item() - val_loss)
            progress_bar.set_postfix(
                loss = f"{val_loss:.2f}"
            )
            progress_bar.update()
        translate(transformer, batch, src_tokenizer, trg_tokenizer)
    progress_bar.close()
    return val_loss


def train_one_epoch(transformer, train_dataloader, loss_fn, optimizer, epoch=1):
    train_loss = 0
    transformer.train()
    device = get_device()
    progress_bar = tqdm(
        total=len(train_dataloader),
        ncols=100,
        desc=f'Training|Epoch-{epoch}'
    )
    for i,batch in enumerate(train_dataloader):
        target = batch["fr_out"].to(device)
        logits = infer(transformer, batch)
        optimizer.zero_grad()
        logits = infer(transformer, batch)
        log_likelihood = log_softmax(logits, dim=-1)
        log_likelihood = log_likelihood.transpose(1,2)
        loss = loss_fn(log_likelihood, target)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + (1/(i+1))*(loss.item()-train_loss)
        progress_bar.set_postfix(
            loss = f"{train_loss:.04f}"
        )
        progress_bar.update()
        
    
    progress_bar.close()
    return train_loss


def train(
        transformer, 
        train_dataloader, 
        val_dataloader, 
        loss_fn, 
        optimizer, 
        scheduler, 
        src_tokenizer,
        trg_tokenizer,
        epochs
    ):
    train_loss = 0
    best_val_loss = 1e10
    for epoch in range(1,epochs+1):
        train_loss = train_loss + \
            (1/epoch)*(train_one_epoch(transformer, train_dataloader, loss_fn, optimizer, epoch) - train_loss)
        val_loss = validate(transformer, val_dataloader, loss_fn, src_tokenizer, trg_tokenizer)
        scheduler.step(val_loss)
        tqdm.write(f"Epoch {epoch}:\n Train Loss: {train_loss} \n Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), "transformer.pt")
        


if __name__=="__main__":
    from model import build_transformer
    from dataset import EnFrDataset
    from torch.nn import NLLLoss
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    train_dataset = EnFrDataset("train", context_len=512)
    val_dataset = EnFrDataset("validation", context_len=512)
    batch_size = 64
    src_vocab_size = train_dataset.en_tokenizer.get_vocab_size()
    trg_vocab_size = train_dataset.fr_tokenizer.get_vocab_size()
    transformer = build_transformer(6, 512, 8, 0.1, src_vocab_size, trg_vocab_size).to(get_device())
    optimizer = Adam(transformer.parameters())
    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = NLLLoss(ignore_index=0)
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size)

    print("Started training")
    src_tokenizer = train_dataset.en_tokenizer
    trg_tokenizer = train_dataset.fr_tokenizer
    train(transformer, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, src_tokenizer, trg_tokenizer, 100)