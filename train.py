import os
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_load import load_dataset, load_config
import click
from tqdm import tqdm
from model.dlrm_prefetcher import DLRMPrefetcher

def train_model(model, data_loader, criterion, optimizer, num_epochs, writer, device):
    model.train()
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            print(f'Epoch {epoch+1}/{num_epochs}, processing batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.squeeze(1), gt_idx.squeeze(1)
            print(table_id_seq.shape, idx_id_seq.shape)
            print(gt_table.shape, gt_idx.shape)
            print()
            optimizer.zero_grad()
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq)
            loss_table = criterion(table_outputs, gt_table)
            loss_idx = criterion(idx_outputs, gt_idx)
            loss = loss_table + loss_idx
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(data_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def validate_model(model, data_loader, criterion, device, fileName):
    model.eval()
    model.to(device)
    val_loss = 0.0
    correct_table = 0
    correct_idx = 0
    total = 0
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f'Validating batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.squeeze(1), gt_idx.squeeze(1)
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq)
            loss_table = criterion(table_outputs, gt_table)
            loss_idx = criterion(idx_outputs, gt_idx)
            loss = loss_table + loss_idx
            val_loss += loss.item()

            _, predicted_table = torch.max(table_outputs, 1)
            _, predicted_idx = torch.max(idx_outputs, 1)
            correct_table += (predicted_table == gt_table).sum().item()
            correct_idx += (predicted_idx == gt_idx).sum().item()
            total += gt_table.size(0)

            for i in range(gt_table.size(0)):
                all_results.append({
                    'gt_table': gt_table[i].item(),
                    'gt_idx': gt_idx[i].item(),
                    'predicted_table': predicted_table[i].item(),
                    'predicted_idx': predicted_idx[i].item()
                })
            torch.cuda.empty_cache()
    
    avg_val_loss = val_loss / len(data_loader)
    accuracy_table = correct_table / total
    accuracy_idx = correct_idx / total
    print(f'Validation Loss: {avg_val_loss:.4f}, Table Accuracy: {accuracy_table:.4f}, Index Accuracy: {accuracy_idx:.4f}')

    results_path = "/home/hli31/S2024_MLSYS/Project-8/tmp/"
    val_results_file = os.path.join(results_path, f"{fileName}_val.txt")
    with open(val_results_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['gt_table']}, {result['gt_idx']}, {result['predicted_table']}, {result['predicted_idx']}\n")
            f.write(f"Validation Loss: {avg_val_loss:.4f}")
            f.write(f"Table Accuracy: {accuracy_table:.4f}")
            f.write(f"Index Accuracy: {accuracy_idx:.4f}\n")

    return avg_val_loss, accuracy_table, accuracy_idx

@click.command()
@click.option(
    "--config",
    type=str,
    default="/home/hli31/S2024_MLSYS/Project-8/configs/prefetcher_transformer.yaml",
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

    hidden_dim = configs['model_configs']['hidden_dim']
    n_heads = configs['model_configs']['n_heads']
    n_layers = configs['model_configs']['n_layers']
    embed_dim = configs['model_configs']['embed_dim']

    lr = configs['training_configs']['learning_rate']
    bs = configs['training_configs']['batch_size']
    epochs = configs['training_configs']['epochs']
    sequence_length = configs['training_configs']['sequence_step']
    prediction_steps = configs['training_configs']['prediction_steps']

    print('Loading training data...')
    train_loader, input_size, _, _ = load_dataset(
        train_set, 
        sequence_length, 
        prediction_steps, 
        bs
    )
    print('Training data loaded.\n')

    print('Loading validation data...')
    val_loader, input_size, _, _ = load_dataset(
        test_set, 
        sequence_length, 
        prediction_steps, 
        bs,
        shuffle=False
    )
    print('Validation data loaded.\n')

    print('Initializing model...')
    model = DLRMPrefetcher(input_size, embed_dim, hidden_dim, n_heads, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Model initialized.\n')

    config_name = os.path.splitext(os.path.basename(config))[0]
    log_dir = os.path.join('runs', 'DLRMPrefetcher', config_name)
    writer = SummaryWriter(log_dir=log_dir)

    print('Starting training...')
    train_model(model, train_loader, criterion, optimizer, epochs, writer, device)
    print('Training completed.\n')

    print('Saving model...')
    model_save_path = os.path.splitext(config)[0] + '.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}\n')

    print('Starting validation...')
    fileName = os.path.splitext(os.path.basename(config))[0]
    validate_model(model, val_loader, criterion, device, fileName)
    print('Validation completed.\n')

    writer.close()


if __name__ == "__main__":
    main()
