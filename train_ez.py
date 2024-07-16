import os
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import click
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_load import load_dataset, load_config
from utils.loss_val import calculate_pair_match, NegativeSamplingLoss
from model.dlrm_prefetcher_ez import DLRMPrefetcher


def train_model0(model, data_loader, table_criterion, idx_criterion, optimizer, num_epochs, writer, device, num_neg_samples):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        epoch_table_loss = 0.0
        epoch_idx_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            print(f'Epoch {epoch+1}/{num_epochs}, processing batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.view(-1, model.output_length), gt_idx.view(-1, model.output_length)

            tgt_table_seq = torch.cat([torch.full((table_id_seq.size(0), 1), model.table_id_vocab - 1).to(device), gt_table[:, :-1]], dim=1)
            tgt_idx_seq = torch.cat([torch.full((idx_id_seq.size(0), 1), model.idx_id_vocab - 1).to(device), gt_idx[:, :-1]], dim=1)
            

            optimizer.zero_grad()
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq, tgt_table_seq, tgt_idx_seq) # (batch_size, seq_len, vocab_size)

            # Positive samples for table_id
            pos_table_output = table_outputs[:, :-1, :] 
            pos_table_target = F.one_hot(gt_table[:, 1:], num_classes=model.table_id_vocab).float() # (batch_size, seq_len, vocab_size)
            
            # Negative samples for table_id
            neg_table_samples = sample_negative_samples(table_id_seq.size(0), num_neg_samples, model.table_id_vocab, device)
            neg_table_output = table_outputs[:, :-1, :].unsqueeze(2).expand(-1, -1, num_neg_samples, -1)
            neg_table_targets = F.one_hot(neg_table_samples, num_classes=model.table_id_vocab).float().unsqueeze(1).expand(-1, model.output_length - 1, -1, -1)

            # Positive samples for idx_id
            pos_idx_output = idx_outputs[:, :-1, :]
            pos_idx_target = F.one_hot(gt_idx[:, 1:], num_classes=model.idx_id_vocab).float()

            # Negative samples for idx_id
            neg_idx_samples = sample_negative_samples(idx_id_seq.size(0), num_neg_samples, model.idx_id_vocab, device)
            neg_idx_output = idx_outputs[:, :-1].unsqueeze(2).expand(-1, -1, num_neg_samples, -1)
            neg_idx_targets = F.one_hot(neg_idx_samples, num_classes=model.idx_id_vocab).float().unsqueeze(1).expand(-1, model.output_length - 1, -1, -1)

            table_loss = table_criterion(pos_table_output, pos_table_target, neg_table_output, neg_table_targets)
            idx_loss = idx_criterion(pos_idx_output, pos_idx_target, neg_idx_output, neg_idx_targets)
            loss = table_loss + idx_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(data_loader)
        avg_table_loss = epoch_table_loss / len(data_loader)
        avg_idx_loss = epoch_idx_loss / len(data_loader)
        writer.add_scalar('Loss/train_total', avg_loss, epoch)
        writer.add_scalar('Loss/table_loss', avg_table_loss, epoch)
        writer.add_scalar('Loss/idx_loss', avg_idx_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def train_model(train_loader, table_vocab_size, idx_vocab_size, embedding_dim, hidden_dim, num_layers, num_epochs):
    model = SequencePredictor(table_vocab_size, idx_vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion_table = nn.CrossEntropyLoss()
    criterion_idx = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        for table_seq, idx_seq, gt_table, gt_idx in train_loader:
            optimizer.zero_grad()
            table_out, idx_out = model(table_seq, idx_seq)
            loss_table = criterion_table(table_out, gt_table)
            loss_idx = criterion_idx(idx_out, gt_idx)
            loss = loss_table + loss_idx
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

file_path = "/home/hli31/S2024_MLSYS/Trace/fbgemm_t856_bs65536_0_trace_551_555.pt"
sequence_length = 32
prediction_ratio = 0.2
batch_size = 256
num_epochs = 5
embedding_dim = 128
hidden_dim = 256
num_layers = 4

train_loader, output_length, table_unq, table_id_map, reverse_table_id_map = load_dataset(file_path, sequence_length, prediction_ratio, batch_size)
train_model(train_loader, table_unq, 10000000, embedding_dim, hidden_dim, num_layers, num_epochs)



@click.command()
@click.option(
    "--config",
    type=str,
    default="./configs/prefetcher_transformer_64_5_2000.yaml",
    help="Path to the configuration file",
)
def main(config):
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", '0')
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    configs = load_config(config)
    print(f'Loaded config from {config}')

    train_set = configs['training_set_path']
    test_set = configs['testing_set_path']

    n_heads = configs['model_configs']['n_heads']
    n_layers = configs['model_configs']['n_layers']
    embed_dim = configs['model_configs']['embed_dim']
    block_size = configs['model_configs']['block_size']

    lr = configs['training_configs']['learning_rate']
    bs = configs['training_configs']['batch_size']
    epochs = configs['training_configs']['epochs']
    input_window_size = configs['training_configs']['input_window_size']
    prediction_ratio = configs['training_configs']['prediction_ratio']

    print('Loading training data...')
    train_loader, output_length, table_unq, idx_unq, _, \
    _, _, _ = load_dataset(
        train_set, 
        input_window_size, 
        prediction_ratio, 
        bs
    )
    print('Training data loaded.\n')

    print('Loading validation data...')
    val_loader, _, _, _, _, \
    reverse_table_id_map, idx_tokenizer, reverse_idx_id_map = load_dataset(
        test_set, 
        input_window_size, 
        prediction_ratio, 
        bs,
        shuffle=False
    )
    print('Validation data loaded.\n')

    print('Initializing model...')
    model = DLRMPrefetcher(table_unq, idx_unq, embed_dim, output_length, block_size, n_heads, n_layers)

    table_criterion = NegativeSamplingLoss(num_neg_samples=5)
    idx_criterion = NegativeSamplingLoss(num_neg_samples=5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Model initialized.\n')

    config_name = os.path.splitext(os.path.basename(config))[0]
    log_dir = os.path.join('runs', 'DLRMPrefetcher', config_name)
    writer = SummaryWriter(log_dir=log_dir)

    print('Starting training...')
    train_model(model, train_loader, table_criterion, idx_criterion, optimizer, epochs, writer, device, num_neg_samples=5)
    print('Training completed.\n')

    print('Saving model...')
    model_save_path = os.path.splitext(config)[0] + '.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}\n')

    print('Starting validation...')
    fileName = os.path.splitext(os.path.basename(config))[0]
    validate_model(model, val_loader, table_criterion, idx_criterion, device, fileName, reverse_table_id_map, reverse_idx_id_map, num_neg_samples=5)
    print('Validation completed.\n')

    writer.close()

if __name__ == "__main__":
    main()