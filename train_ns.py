import os
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import click
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_load_ez import load_dataset, load_config
from model.dlrm_prefetcher_ns import DLRMPrefetcher
from utils.loss_val import calculate_pair_match, JaccardLoss


def train_model(model, data_loader, table_criterion, idx_criterion, optimizer, num_epochs, writer, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        epoch_table_loss = 0.0
        epoch_idx_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            print(f'Epoch {epoch+1}/{num_epochs}, processing batch {batch_idx+1}/{len(data_loader)}')
            #TODO
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            #table_seq, idx_seq, gt_table, gt_idx = table_seq.to(device), idx_seq.to(device), gt_table.to(device), gt_idx.to(device)
            
            optimizer.zero_grad()
            table_out, idx_out = model(table_id_seq, idx_id_seq, gt_table, gt_idx)

            table_loss = table_criterion(table_out, gt_table)
            idx_loss = idx_criterion(idx_out, gt_idx)

            loss = table_loss + idx_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_table_loss += table_loss.item()
            epoch_idx_loss += idx_loss.item()
        torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(data_loader)
        avg_table_loss = epoch_table_loss / len(data_loader)
        avg_idx_loss = epoch_idx_loss / len(data_loader)
        writer.add_scalar('Loss/train_total', avg_loss, epoch)
        writer.add_scalar('Loss/table_loss', avg_table_loss, epoch)
        writer.add_scalar('Loss/idx_loss', avg_idx_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def validate_model(model, val_loader, table_criterion, idx_criterion, device, log_file, reverse_table_id_map):
    model.eval()
    model.to(device)
    val_loss = 0.0
    val_table_loss = 0.0
    val_idx_loss = 0.0
    pair_match_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f'Validation processing batch {batch_idx+1}/{len(val_loader)}')

            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]

            table_out, idx_out = model(table_id_seq, idx_id_seq, gt_table, gt_idx)
            
            table_loss = table_criterion(table_out, gt_table)
            idx_loss = idx_criterion(idx_out, gt_idx)
            
            loss = table_loss + idx_loss
            val_loss += loss.item()
            val_table_loss += table_loss.item()
            val_idx_loss += idx_loss.item()

            pair_match_accuracy += calculate_pair_match(torch.cat((table_out, idx_out), dim=-1), torch.cat((gt_table, gt_idx), dim=-1))

        val_loss /= len(val_loader)
        val_table_loss /= len(val_loader)
        val_idx_loss /= len(val_loader)
        pair_match_accuracy /= len(val_loader)

        print(f'Validation Loss: {val_loss:.4f}, Table Loss: {val_table_loss:.4f}, IDX Loss: {val_idx_loss:.4f}, Pair Match Accuracy: {pair_match_accuracy:.4f}')

        with open(log_file, 'a') as f:
            f.write(f'Validation Loss: {val_loss:.4f}, Table Loss: {val_table_loss:.4f}, IDX Loss: {val_idx_loss:.4f}, Pair Match Accuracy: {pair_match_accuracy:.4f}\n')


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

    lr = configs['training_configs']['learning_rate']
    bs = configs['training_configs']['batch_size']
    epochs = configs['training_configs']['epochs']
    input_window_size = configs['training_configs']['input_window_size']
    prediction_ratio = configs['training_configs']['prediction_ratio']

    print('Loading training data...')
    train_loader, output_length, table_unq, _, _ = load_dataset(
        train_set, 
        input_window_size, 
        prediction_ratio, 
        bs
    )
    print('Training data loaded.\n')

    print('Loading validation data...')
    val_loader, output_length, table_unq_val, table_id_map_val, reverse_table_id_map_val = load_dataset(
        test_set, 
        input_window_size, 
        prediction_ratio, 
        bs,
        shuffle=False
    )
    print('Validation data loaded.\n')

    print('Initializing model...')
    model = DLRMPrefetcher(table_unq, 100000, embed_dim, output_length, n_heads, n_layers)

    table_criterion = JaccardLoss()
    idx_criterion = JaccardLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Model initialized.\n')

    config_name = os.path.splitext(os.path.basename(config))[0]
    log_dir = os.path.join('runs', 'DLRMPrefetcher', config_name)
    writer = SummaryWriter(log_dir=log_dir)

    print('Starting training...')
    train_model(model, train_loader, table_criterion, idx_criterion, optimizer, epochs, writer, device)
    print('Training completed.\n')

    print('Saving model...')
    model_save_path = os.path.splitext(config)[0] + '.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}\n')

    print('Starting validation...')
    fileName = os.path.splitext(os.path.basename(config))[0]
    validate_model(model, val_loader, table_criterion, idx_criterion, device, fileName, reverse_table_id_map)
    print('Validation completed.\n')

    writer.close()

if __name__ == "__main__":
    main()
