import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
import warnings

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang] 

def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(speacial_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"], min_frequency=2)   
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds = load_dataset("Helsinki-NLP/opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    #build tokenizers
    tokenizer_src = build_tokenizer(config, ds, config["lang_src"])
    tokenizer_tgt = build_tokenizer(config, ds, config["lang_tgt"])

    #90% for training, 10% for validation
    train_ds_size = int(0.9*len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length src: {max_len_src}, Max length tgt: {max_len_tgt}")

    train_data_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(config, vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    #define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_data_loader, val_data_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("<PAD>"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) # [batch_size, seq_len]
            decoder_input = batch["decoder_input"].to(device) # [batch_size, seq_len]
            encoder_mask = batch["encoder_mask"].to(device) # [batch_size, 1, 1, seq_len]
            decoder_mask = batch["decoder_mask"].to(device) # [batch_size, 1, seq_len, seq_len]

            #run the tensors through the transformer
            encoder_output = model.encoder(encoder_input, encoder_mask) # [batch_size, seq_len, d_model]
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask) # [batch_size, seq_len, d_model]
            proj_output = model.proj(decoder_output) # [batch_size, seq_len, vocab_tgt_len]

            label = batch["label"].to(device) # [batch_size, seq_len]   
            
            #(batch_size, seq_len, vocab_tgt_len) -> (batch_size*seq_len, vocab_tgt_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            #log the loss
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            #backpropagate the loss
            loss.backward()
            #update the weights
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        #save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


