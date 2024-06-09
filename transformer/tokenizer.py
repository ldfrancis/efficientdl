import os

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import load_dataset
hf_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

def build_tokenizer(lang, folder="tokenizers"):
    tokenizer_file = f"{folder}/{lang}.json"
    if os.path.exists(tokenizer_file):
        return Tokenizer.from_file(tokenizer_file)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    pairs = (hf_dataset["train"]["translation"]+hf_dataset["validation"]["translation"])
    corpus = (
        [pair[lang] for pair in pairs[i:i+1000]] 
        for i in range(0, len(pairs), 1000)
    )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<pad>", "<sos>", "<eos>"])
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(tokenizer_file)
    
    return tokenizer
    