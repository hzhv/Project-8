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
from model.dlrm_prefetcher_ns import DLRMPrefetcher


def train_model(model, data_loader, table_criterion, idx_criterion, optimizer, num_epochs, writer, device, num_neg_samples):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            print(f'Epoch {epoch+1}/{num_epochs}, processing batch {batch_idx+1}/{len(data_loader)}')
            table_id_seq, idx_id_seq, gt_table, gt_idx = [x.to(device) for x in batch]
            gt_table, gt_idx = gt_table.view(-1, model.output_length), gt_idx.view(-1, model.output_length)
            print("Table ID gt shape:", gt_table.shape)
            print("Idx ID gt shape:", gt_idx.shape)

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
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def validate_model(model, data_loader, table_criterion, idx_criterion, device, fileName, reverse_table_id_map, reverse_idx_id_map, num_neg_samples):
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
            gt_table, gt_idx = gt_table.view(-1, model.output_length), gt_idx.view(-1, model.output_length)

            tgt_table_seq = torch.cat([torch.full((table_id_seq.size(0), 1), model.table_id_vocab - 1).to(device), gt_table[:, :-1]], dim=1)
            tgt_idx_seq = torch.cat([torch.full((idx_id_seq.size(0), 1), model.idx_id_vocab - 1).to(device), gt_idx[:, :-1]], dim=1)

            table_output_embeds, idx_output_embeds = model(table_id_seq, idx_id_seq, tgt_table_seq, tgt_idx_seq)

            pos_table_output = table_output_embeds[:, :-1, :]
            pos_table_target = F.one_hot(gt_table[:, 1:], num_classes=model.table_id_vocab).float()

            neg_table_samples = sample_negative_samples(table_id_seq.size(0), num_neg_samples, model.table_id_vocab, device)
            neg_table_output = table_output_embeds[:, :-1].unsqueeze(2).expand(-1, -1, num_neg_samples, -1)
            neg_table_targets = F.one_hot(neg_table_samples, num_classes=model.table_id_vocab).float().unsqueeze(1).expand(-1, model.output_length - 1, -1, -1)

            pos_idx_output = idx_output_embeds[:, :-1]
            pos_idx_target = F.one_hot(gt_idx[:, 1:], num_classes=model.idx_id_vocab).float()

            neg_idx_samples = sample_negative_samples(idx_id_seq.size(0), num_neg_samples, model.idx_id_vocab, device)
            neg_idx_output = idx_output_embeds[:, :-1].unsqueeze(2).expand(-1, -1, num_neg_samples, -1)
            neg_idx_targets = F.one_hot(neg_idx_samples, num_classes=model.idx_id_vocab).float().unsqueeze(1).expand(-1, model.output_length - 1, -1, -1)

            table_loss = table_criterion(pos_table_output, pos_table_target, neg_table_output, neg_table_targets)
            idx_loss = idx_criterion(pos_idx_output, pos_idx_target, neg_idx_output, neg_idx_targets)
            loss = table_loss + idx_loss
            val_loss += loss.item()

            _, predicted_table = torch.max(table_output_embeds, dim=2)  # Changed dimension to 2
            _, predicted_idx = torch.max(idx_output_embeds, dim=2)  # Changed dimension to 2

            predicted_table_original = torch.tensor([reverse_table_id_map[v.item()] for v in predicted_table.view(-1)]).view(predicted_table.size())
            predicted_idx_original = torch.tensor([reverse_idx_id_map[v.item()] for v in predicted_idx.view(-1)]).view(predicted_idx.size())

            gt_table_original = torch.tensor([reverse_table_id_map[v.item()] for v in gt_table.view(-1)]).view(gt_table.size())
            gt_idx_original = torch.tensor([reverse_idx_id_map[v.item()] for v in gt_idx.view(-1)]).view(gt_idx.size())

            predicted_pairs = torch.stack((predicted_table_original, predicted_idx_original), dim=2)
            gt_pairs = torch.stack((gt_table_original, gt_idx_original), dim=2)

            accuracy = calculate_pair_match(predicted_pairs, gt_pairs)
            total_samples += 1
            total_accuracy += accuracy

            for i in range(gt_table.size(0)):
                for j in range(gt_table.size(1)):
                    all_results.append({
                        'gt_table': gt_pairs[i, j, 0].item(),
                        'gt_idx': gt_pairs[i, j, 1].item(),
                        'predicted_table': predicted_pairs[i, j, 0].item(),
                        'predicted_idx': predicted_pairs[i, j, 1].item()
                    })
            torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(data_loader)
    average_accuracy = total_accuracy / total_samples

    print(f'Validation Loss: {avg_val_loss:.4f}, Pair Match Accuracy: {average_accuracy:.4f}')

    results_path = "./val_results/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    val_results_file = os.path.join(results_path, f"{fileName}_val.txt")
    with open(val_results_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['gt_table']}, {result['gt_idx']}, {result['predicted_table']}, {result['predicted_idx']}\n")
        f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
        f.write(f"Pair Match Accuracy: {average_accuracy:.4f}\n")

    return avg_val_loss, average_accuracy


def sample_negative_samples(batch_size, num_neg_samples, vocab_size, device):
    neg_samples = torch.randint(0, vocab_size, (batch_size, num_neg_samples)).to(device)
    return neg_samples

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
    print('Training data loaded.\n')

    print('Loading validation data...')
    val_loader, _, _, _, reverse_table_id_map, idx_tokenizer = load_dataset(
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
    validate_model(model, val_loader, table_criterion, idx_criterion, device, fileName, reverse_table_id_map, idx_tokenizer, num_neg_samples=5)
    print('Validation completed.\n')

    writer.close()

if __name__ == "__main__":
    main()
