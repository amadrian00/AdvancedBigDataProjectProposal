import os
import torch
from model import GCN
from scipy.stats import tstd
from train_eval import train, evaluate
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

if __name__ == '__main__':
    add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare BACE dataset
    dataset_bace, split_bace, idx_bace, y_bace = prepare_dataset("ogbg-molbace")

    train_loader_bace = create_data_loader(dataset_bace, idx_bace, shuffle=True)
    test_loader_bace = create_data_loader(dataset_bace, split_bace["test"])

    # Prepare training dataset
    dataset_bbbp, split_bbbp, idx_bbbp, y_bbbp = prepare_dataset("ogbg-molbbbp", check_nan=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(torch.zeros_like(y_bbbp), y_bbbp)):
        print(f'\n--- Fold {fold + 1} ---')

        train_subset = dataset_bbbp[idx_bbbp[train_idx]]
        val_subset = dataset_bbbp[idx_bbbp[val_idx]]

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64)

        #model = GCN(9, dataset.y.shape[1]).to(device)
        model = GCN(9, 1).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()

        train(model, train_loader, optimizer, criterion, device, num_epochs=100)

        for name, param in model.named_parameters():
            if not name.startswith('lin'):
                param.requires_grad = False

        model.lin.reset_parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        train(model, train_loader_bace, optimizer, criterion, device, num_epochs=15)

        metrics = evaluate(model, test_loader_bace, device)
        all_metrics.append(metrics)
        print(metrics)

    keys = all_metrics[0].keys()

    print("\n--- 5 Folds Average ---")
    print(f"{'Metric':<15}{'Mean':>10}{'Std':>10}")
    print("-" * 35)
    for key in keys:
        values = [m[key] for m in all_metrics]
        mean_val = sum(values) / len(values)
        std_val = tstd(values)
        print(f"{key:<15}{mean_val:>10.4f}{std_val:>10.4f}")