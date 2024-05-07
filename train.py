import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import click

from model.prefetcher import PrefetcherTransformer
from utils.dataset_load import load_dataset, load_config


@click.command()
@click.option(
    "--config",
    type=str,
    default="/home/hli31/S2024_MLSYS/Project-8/configs/prefetcher_transformer.yaml",
    help="Path to the configuration file",
)
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_config(config)

    train_set = configs['training_set_path']
    test_set = configs['testing_set_path']
    
    lr = configs['training_configs']['learning_rate']
    bs = configs['training_configs']['batch_size']
    epochs = configs['training_configs']['epochs']
    sequence_length = configs['training_configs']['sequence_length']
    prediction_steps = configs['training_configs']['prediction_steps']
    alpha = configs['training_configs']['loss_weights']['alpha']
    beta = configs['training_configs']['loss_weights']['beta']

    train_loader, num_tables, num_idx = load_dataset(train_set, sequence_length, prediction_steps, bs)
    training_data_params = {
        'num_classes_table': num_tables,
        'num_classes_idx': num_idx
    }
    test_loader, _, _= load_dataset(test_set, sequence_length, prediction_steps, bs)
    
    model_configs = {**configs['model_configs'], **training_data_params} 
    model = PrefetcherTransformer(**model_configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir='/home/hli31/S2024_MLSYS/Project-8/logs')
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device, writer, alpha, beta, epoch)
        logging.info(f'Epoch {epoch}, Train Loss: {train_loss}')
        
        accuracy = test(model, test_loader, device, writer, epoch)
        logging.info(f'Test Accuracy: {accuracy}')
    
    writer.close()

#####################################################################################
# train_and_test
def train(model, data_loader, optimizer, device, writer, alpha, beta, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training')
    for i, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt_table, tgt_idx = tgt.to(device)
        optimizer.zero_grad()

        pred_table, pred_idx = model(src, tgt)
        loss = compute_loss(pred_table, tgt_table, pred_idx, tgt_idx, alpha, beta)
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(data_loader) + i)
    return total_loss / len(data_loader)

def test(model, data_loader, device, writer, epoch):
    model.eval()
    total_accuracy_strict, total_accuracy_part, total_accuracy_ave = 0.0, 0.0, 0.0
    total_samples = 0
    with torch.no_grad():
        for src, (tgt_table, tgt_idx) in data_loader:
            src = src.to(device)
            tgt_table, tgt_idx = tgt_table.to(device), tgt_idx.to(device)
            pred_table, pred_idx = model(src)
            
            accuracy_strict = acc_strict((pred_table, pred_idx), (tgt_table, tgt_idx))
            accuracy_table, accuracy_idx = acc_part((pred_table, pred_idx), (tgt_table, tgt_idx))
            accuracy_ave = acc_ave((pred_table, pred_idx), (tgt_table, tgt_idx))
            
            total_accuracy_strict += accuracy_strict
            total_accuracy_part += (accuracy_table + accuracy_idx) / 2 
            total_accuracy_ave += accuracy_ave
            total_samples += 1

    final_accuracy_strict = total_accuracy_strict / total_samples
    final_accuracy_part = total_accuracy_part / total_samples
    final_accuracy_ave = total_accuracy_ave / total_samples

    writer.add_scalar('Test/Accuracy_Strict', final_accuracy_strict, epoch)
    writer.add_scalar('Test/Accuracy_Part', final_accuracy_part, epoch)
    writer.add_scalar('Test/Accuracy_Ave', final_accuracy_ave, epoch)
    
    return final_accuracy_strict, final_accuracy_part, final_accuracy_ave

#####################################################################################
def compute_loss(pred_table, true_table, pred_idx, true_idx, alpha, beta):
    criterion_table = nn.CrossEntropyLoss()
    criterion_idx = nn.CrossEntropyLoss()

    loss_table = criterion_table(pred_table, true_table)
    loss_idx = criterion_idx(pred_idx, true_idx)
    
    total_loss = alpha * loss_table + beta * loss_idx
    return total_loss

def acc_strict(preds, labels):
    pred_table_ids, pred_idx_ids = preds
    true_table_ids, true_idx_ids = labels
    
    correct_table = (pred_table_ids == true_table_ids)
    correct_idx = (pred_idx_ids == true_idx_ids)
    
    strict_correct = (correct_table & correct_idx).all().item()
    
    return strict_correct

def acc_part(preds, labels):
    pred_table_ids, pred_idx_ids = preds
    true_table_ids, true_idx_ids = labels

    correct_table = (pred_table_ids == true_table_ids).float().sum()
    correct_idx = (pred_idx_ids == true_idx_ids).float().sum()
    
    total = pred_table_ids.size(0)
    
    acc_table = correct_table / total
    acc_idx = correct_idx / total
    
    return acc_table.item(), acc_idx.item() 

def acc_ave(preds, labels):
    pred_table_ids, pred_idx_ids = preds
    true_table_ids, true_idx_ids = labels
    correct_table = (pred_table_ids == true_table_ids).float().sum()
    correct_idx = (pred_idx_ids == true_idx_ids).float().sum()
    total = len(pred_table_ids)
    acc_table = correct_table / total
    acc_idx = correct_idx / total
    return (acc_table + acc_idx) / 2


if __name__ == '__main__':
    main()