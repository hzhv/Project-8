import os
import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_load import load_dataset, load_config
from utils.loss_val import calculate_pair_match
import click
from tqdm import tqdm
from model.dlrn_prefetcher2 import DLRMPrefetcher


def train_model(model, data_loader, criterion, optimizer, num_epochs, writer, device):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            print(f'Epoch {epoch+1}/{num_epochs}, processing batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.view(-1), gt_idx.view(-1)

            print(f"table_id_seq: min={table_id_seq.min()}, max={table_id_seq.max()}")
            print(f"idx_id_seq: min={idx_id_seq.min()}, max={idx_id_seq.max()}\n")
            
            optimizer.zero_grad()
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq) # output these at the same time

            ##### Test code:
            print(f'table_outputs shape: {table_outputs.shape}')  # [batch_size, prediction_steps, num_classes]
            print(f'idx_outputs shape: {idx_outputs.shape}')      # [batch_size, prediction_steps, num_classes]
            print(f'gt_table shape: {gt_table.shape}')            # [batch_size, prediction_steps]
            print(f'gt_idx shape: {gt_idx.shape}\n')              # [batch_size, prediction_steps]
            
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

def validate_model(model, data_loader, criterion, device, fileName, reverse_table_id_map, reverse_idx_id_map):
    model.eval()
    model.to(device)
    val_loss = 0.0
    total_accuracy = 0
    total_samples = 0
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f'Validating batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.view(-1), gt_idx.view(-1)
            ## Debugging:
            print("##### Shape check:")
            print("gt ready", gt_table, gt_idx, "\nand their shape:", gt_table.shape, gt_idx.shape)
            # print("the original gt:", gt_table_original, gt_idx_original, "and their shape:", gt_table_original.shape, gt_idx_original.shape)
            
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq)
            print("output shape:", table_outputs.shape, idx_outputs.shape,"\n")

            ## Error checking:
            if gt_table.min() < 0 or gt_table.max() >= table_outputs.size(1):
                print(f"Error: gt_table contains invalid values")
            if gt_idx.min() < 0 or gt_idx.max() >= idx_outputs.size(1):
                print(f"Error: gt_idx contains invalid values")

            loss_table = criterion(table_outputs, gt_table)
            loss_idx = criterion(idx_outputs, gt_idx)
            loss = loss_table + loss_idx
            val_loss += loss.item()

            _, predicted_table = torch.max(table_outputs, 1)
            _, predicted_idx = torch.max(idx_outputs, 1)
            
            predicted_table_original = torch.tensor([reverse_table_id_map[v.item()] for v in predicted_table]).view(predicted_table.size())
            predicted_idx_original = torch.tensor([reverse_idx_id_map[v.item()] for v in predicted_idx]).view(predicted_idx.size())

            gt_table_original = torch.tensor([reverse_table_id_map[v.item()] for v in gt_table]).view(gt_table.size())
            gt_idx_original = torch.tensor([reverse_idx_id_map[v.item()] for v in gt_idx]).view(gt_idx.size())

            predicted_pairs = torch.stack((predicted_table_original, predicted_idx_original), dim=1)
            gt_pairs = torch.stack((gt_table_original, gt_idx_original), dim=1)

            accuracy = calculate_pair_match(predicted_pairs, gt_pairs)
            ## Debugging:
            # print("the original gt:", gt_table_original, gt_idx_original, "and their shape:", gt_table_original.shape, gt_idx_original.shape)
            # print("let's see what stacked orginal gt looks like:", gt_pairs)
            print("Current Accuracy: ", accuracy)
            ## End
            total_samples += 1
            total_accuracy += accuracy

            for i in range(gt_table.size(0)):
                all_results.append({
                    'gt_table': gt_pairs[i, 0].item(),
                    'gt_idx': gt_pairs[i, 1].item(),
                    'predicted_table': predicted_pairs[i, 0].item(),
                    'predicted_idx': predicted_pairs[i, 1].item()
                })
            torch.cuda.empty_cache()
    
    avg_val_loss = val_loss / len(data_loader)
    average_accuracy = total_accuracy / total_samples

    print(f'Validation Loss: {avg_val_loss:.4f}, Pair Match Accuracy: {average_accuracy:.4f}')

    results_path = "./Project-8/tmp/"
    val_results_file = os.path.join(results_path, f"{fileName}_val.txt")
    with open(val_results_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['gt_table']}, {result['gt_idx']}, {result['predicted_table']}, {result['predicted_idx']}\n")
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
        f.write(f"Pair Match Accuracy: {average_accuracy:.4f}\n")

    return avg_val_loss, average_accuracy


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

    hidden_dim = configs['model_configs']['hidden_dim']
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
    train_loader, output_length, table_unq, idx_unq, _, _ = load_dataset(
        train_set, 
        input_window_size, 
        prediction_ratio, 
        bs
    )
    # print("Unique table_id: ", table_unq)
    # print("Unique idx_id: ", idx_unq)
    print('Training data loaded.\n')
    ############################################################################################################
    print('Loading validation data...')
    val_loader, output_length, _, _, table_id_map_val, idx_id_map_val = load_dataset(
        test_set, 
        input_window_size, 
        prediction_ratio, 
        bs,
        shuffle=False
    )
    print('Validation data loaded.\n')
    ############################################################################################################
    print('Initializing model...')
    model = DLRMPrefetcher(table_unq, idx_unq, embed_dim, output_length, block_size, n_heads, n_layers)

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
    reverse_table_id_map = {v: k for k, v in table_id_map_val.items()}
    reverse_idx_id_map = {v: k for k, v in idx_id_map_val.items()}
    print(f"Reverse table_id map: {reverse_table_id_map}")
    print(f"Reverse idx_id map: {reverse_idx_id_map}")
    validate_model(model, val_loader, criterion, device, fileName, reverse_table_id_map, reverse_idx_id_map)
    print('Validation completed.\n')

    writer.close()


if __name__ == "__main__":
    main()
