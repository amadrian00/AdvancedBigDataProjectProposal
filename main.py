import os
import torch
from model import GCN
from train_eval import train, evaluate
from scipy.stats import tstd, ttest_rel
from torch_geometric.loader import DataLoader
from torch.serialization import add_safe_globals
from sklearn.model_selection import StratifiedKFold
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_dataset(dataset_name, label_col=0, check_nan=False):
    dataset = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()

    # Concatenate train and valid indices and corresponding labels
    idx = torch.concat((split_idx['train'], split_idx['valid']))
    y = torch.concat((dataset.y[split_idx["train"], label_col],
                      dataset.y[split_idx["valid"], label_col]))

    if not check_nan:
        valid_mask = ~torch.isnan(y)
        idx = idx[valid_mask]
        y = y[valid_mask]

    return dataset, split_idx, idx, y

def create_data_loader(dataset, indices, batch_size=64, shuffle=False):
    return DataLoader(dataset[indices], batch_size=batch_size, shuffle=shuffle)

def train_test_kfold(dataset, y, train_id, test_loader_ext, train_loader_ft=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_metrics = []
    for fold, (train_idx, _) in enumerate(skf.split(torch.zeros_like(y), y)):
        print(f'\n--- Fold {fold + 1} ---')

        train_loader = create_data_loader(dataset, train_id[train_idx], shuffle=True)

        model = GCN(9, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        train(model, train_loader, optimizer, criterion, device, num_epochs=100)

        if train_loader_ft:
            for name, param in model.named_parameters():
                if not name.startswith('lin'):
                    param.requires_grad = False

            model.lin.reset_parameters()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

            train(model, train_loader_ft, optimizer, criterion, device, num_epochs=100)

        metrics = evaluate(model, test_loader_ext, device)
        all_metrics.append(metrics)
        print(metrics)
    return all_metrics

if __name__ == '__main__':
    add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = [42, 84, 126, 168, 210]

    all_metrics_bace_all_runs = []
    all_metrics_bbbp_all_runs = []

    for seed in seeds:
        print(f"\nRunning with seed: {seed}")
        set_seed(seed)

        ####################################################################################################
        ###     This test loader MUST not be changed and SHOULD be used for reporting final results.     ###
        ####################################################################################################
        dataset_bace, split_bace, idx_bace, y_bace = prepare_dataset("ogbg-molbace")
        test_loader_bace = create_data_loader(dataset_bace, split_bace["test"])
        ####################################################################################################
        ####################################################################################################

        train_loader_bace = create_data_loader(dataset_bace, idx_bace, shuffle=True)
        all_metrics_bace = train_test_kfold(dataset_bace, y_bace, idx_bace, test_loader_bace)
        all_metrics_bace_all_runs.append(all_metrics_bace)

        dataset_bbbp, split_bbbp, idx_bbbp, y_bbbp = prepare_dataset("ogbg-molbbbp", check_nan=True)
        all_metrics_bbbp = train_test_kfold(dataset_bbbp, y_bbbp, idx_bbbp, test_loader_bace, train_loader_ft=train_loader_bace)
        all_metrics_bbbp_all_runs.append(all_metrics_bbbp)

    flatten = lambda l: [item for sublist in l for item in sublist]
    flat_bace = flatten(all_metrics_bace_all_runs)
    flat_bbbp = flatten(all_metrics_bbbp_all_runs)

    print("\n--- 5 Folds Average + Paired t-test ---")
    print(
        f"{'Metric':<15}{'Mean BACE':>12}{'Std BACE':>12}{'Mean BBBP':>15}{'Std BBBP':>12}{'% Inc.':>15}{'p-value':>12}")
    print("-" * 99)

    for key in flat_bace[0].keys():
        values_bace = [m[key] for m in flat_bace]
        values_bbbp = [m[key] for m in flat_bbbp]

        mean_bace = sum(values_bace) / len(values_bace)
        std_bace = tstd(values_bace)

        mean_bbbp = sum(values_bbbp) / len(values_bbbp)
        std_bbbp = tstd(values_bbbp)

        t_stat, p_val = ttest_rel(values_bbbp, values_bace)

        try:
            pct_inc = ((mean_bbbp - mean_bace) / abs(mean_bace)) * 100
        except ZeroDivisionError:
            pct_inc = float('nan')

        print(f"{key:<15}{mean_bace:>12.4f}{std_bace:>12.4f}{mean_bbbp:>15.4f}{std_bbbp:>12.4f}{pct_inc:>15.2f}{p_val:>12.4f}")
