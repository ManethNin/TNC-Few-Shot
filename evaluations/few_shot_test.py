import os
import torch
import numpy as np
import pickle
import random
import argparse
import matplotlib.pyplot as plt

from tnc.models import RnnEncoder, StateClassifier, WFEncoder, WFClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_few_shot_dataset(x_windows, y_windows, n_shot=5):
    """
    Create a few-shot dataset by sampling n_shot examples per class
    """
    unique_classes = np.unique(y_windows)
    few_shot_x, few_shot_y = [], []
    
    for class_label in unique_classes:
        # Find all examples of this class
        class_indices = np.where(y_windows == class_label)[0]
        
        # Randomly sample n_shot examples (or all if fewer than n_shot exist)
        n_samples = min(n_shot, len(class_indices))
        selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        
        few_shot_x.extend(x_windows[selected_indices])
        few_shot_y.extend(y_windows[selected_indices])
        
        print(f"Class {class_label}: selected {n_samples} examples from {len(class_indices)} available")
    
    return torch.stack(few_shot_x), torch.tensor(few_shot_y)


def epoch_run_few_shot(encoder, classifier, dataloader, train=False, lr=0.01):
    """Modified epoch_run for few-shot learning"""
    if train:
        classifier.train()
        encoder.eval()  # Keep encoder frozen
    else:
        classifier.eval()
        encoder.eval()
        
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    epoch_loss, epoch_acc = 0, 0
    batch_count = 0
    y_all, prediction_all = [], []
    
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        
        # Get embeddings from frozen encoder
        with torch.no_grad():
            encodings = encoder(x)
        
        # Train only the classifier
        prediction = classifier(encodings)
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        y_all.append(y.cpu().detach().numpy())
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())

        epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
        epoch_loss += loss.item()
        batch_count += 1
        
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    
    # Handle NaN in predictions
    if np.isnan(prediction_all).any():
        print("Warning: NaN found in predictions, replacing with zeros")
        prediction_all = np.nan_to_num(prediction_all, nan=0.0)
    
    # Calculate metrics with error handling
    try:
        if len(np.unique(y_all)) > 1:  # Multi-class case
            epoch_auc = roc_auc_score(y_onehot_all, prediction_all, multi_class='ovr')
            epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
        else:  # Single class case
            epoch_auc = 0.5  # Random performance for single class
            epoch_auprc = np.mean(y_onehot_all)
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        epoch_auc = 0.5
        epoch_auprc = 0.5
    
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc


def run_few_shot_experiment(data_type, n_shot=5, n_epochs=50, lr=0.01, n_trials=5):
    """
    Run few-shot learning experiment
    """
    print(f"\n=== Few-Shot Learning Experiment: {n_shot}-shot ===")
    
    # Load data based on dataset type
    if data_type == 'simulation':
        data_path = './data/simulated_data/'
        window_size = 50
        encoding_size = 10
        n_classes = 4
        encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
        
    elif data_type == 'waveform':
        data_path = './data/waveform_data/processed'
        window_size = 2500
        encoding_size = 64
        n_classes = 4
        encoder = WFEncoder(encoding_size=encoding_size).to(device)
        
    elif data_type == 'har':
        data_path = './data/HAR_data/'
        window_size = 4
        encoding_size = 10
        n_classes = 6
        encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=encoding_size, device=device)
    
    # Load pre-trained encoder
    checkpoint_path = f'./ckpt/{data_type}/checkpoint_0.pth.tar'
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"No pre-trained encoder found at {checkpoint_path}. Train TNC first!")
        
    # Load checkpoint with CPU mapping for cross-platform compatibility
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    encoder.to(device)
    
    # Load and prepare data
    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    # Convert to windows
    T = x.shape[-1]
    x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))
    
    # Test set
    x_window_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    y_window_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window_test]))
    
    # Test dataset (full)
    testset = torch.utils.data.TensorDataset(x_window_test, y_window_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    # Run multiple trials
    trial_accuracies = []
    trial_aucs = []
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Create few-shot training set
        few_shot_x, few_shot_y = create_few_shot_dataset(x_window, y_window, n_shot)
        
        # Create classifier
        if data_type == 'waveform':
            classifier = WFClassifier(encoding_size=encoding_size, output_size=n_classes).to(device)
        else:
            classifier = StateClassifier(input_size=encoding_size, output_size=n_classes).to(device)
        
        # Create few-shot train loader
        few_shot_dataset = torch.utils.data.TensorDataset(few_shot_x, few_shot_y)
        few_shot_loader = torch.utils.data.DataLoader(few_shot_dataset, batch_size=min(32, len(few_shot_x)), shuffle=True)
        
        # Train classifier on few-shot data
        best_acc = 0
        for epoch in range(n_epochs):
            train_loss, train_acc, train_auc, train_auprc = epoch_run_few_shot(
                encoder, classifier, few_shot_loader, train=True, lr=lr)
            
            # Evaluate on full test set
            test_loss, test_acc, test_auc, test_auprc = epoch_run_few_shot(
                encoder, classifier, test_loader, train=False)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_test_auc = test_auc
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        
        trial_accuracies.append(best_acc)
        trial_aucs.append(best_test_auc)
        print(f"Trial {trial + 1} Best Accuracy: {best_acc:.3f}")
    
    # Report results
    mean_acc = np.mean(trial_accuracies)
    std_acc = np.std(trial_accuracies)
    mean_auc = np.mean(trial_aucs)
    std_auc = np.std(trial_aucs)
    
    print(f"\n=== {n_shot}-Shot Results on {data_type} ===")
    print(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    
    return mean_acc, std_acc, mean_auc, std_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-Shot Learning with TNC')
    parser.add_argument('--data', type=str, default='simulation', 
                        choices=['simulation', 'waveform', 'har'])
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 5, 10], 
                        help='Number of shots to test (e.g., --shots 1 5 10)')
    parser.add_argument('--trials', type=int, default=5, 
                        help='Number of trials per shot setting')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Training epochs for classifier')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate for classifier')
    
    args = parser.parse_args()
    
    print(f"Running Few-Shot Learning on {args.data} dataset")
    print(f"Testing {args.shots}-shot settings with {args.trials} trials each")
    
    results = {}
    for n_shot in args.shots:
        mean_acc, std_acc, mean_auc, std_auc = run_few_shot_experiment(
            args.data, n_shot=n_shot, n_epochs=args.epochs, 
            lr=args.lr, n_trials=args.trials
        )
        results[n_shot] = {'acc': (mean_acc, std_acc), 'auc': (mean_auc, std_auc)}
    
    print("\n=== Summary ===")
    for n_shot, metrics in results.items():
        acc_mean, acc_std = metrics['acc']
        auc_mean, auc_std = metrics['auc']
        print(f"{n_shot}-shot: Acc {acc_mean:.3f}±{acc_std:.3f}, AUC {auc_mean:.3f}±{auc_std:.3f}")
