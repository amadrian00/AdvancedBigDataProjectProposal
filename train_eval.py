import torch
from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryConfusionMatrix, BinaryAUROC

def compute_metrics(y_true, y_probs):
    acc = BinaryAccuracy(threshold=0.5)
    f1 = BinaryF1Score(threshold=0.5)
    recall = BinaryRecall(threshold=0.5)
    cm = BinaryConfusionMatrix(threshold=0.5)
    aur = BinaryAUROC()

    acc.update(y_probs, y_true)
    f1.update(y_probs, y_true)
    recall.update(y_probs, y_true)
    cm.update(y_probs, y_true)
    aur.update(y_probs, y_true)

    tn, fp, fn, tp = cm.compute().flatten()

    spec = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)

    return {
        "accuracy": acc.compute().item(),
        "sensitivity": recall.compute().item(),
        "specificity": spec.item(),
        "f1": f1.compute().item(),
        "roc": aur.compute().item(),
    }

def train(model, loader, optimizer, criterion, device, num_epochs):
    progress_bar = tqdm(range(num_epochs), desc="Training", leave=True)

    for _ in progress_bar:
        model.train()
        total_loss = 0.0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            loss = criterion(out, data.y[:,0].float().view_as(out))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        progress_bar.set_postfix(loss=avg_loss)

    return avg_loss

def evaluate(model, loader, device):
    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            probs = torch.sigmoid(out)
            y_probs.append(probs.cpu())
            y_true.append(data.y[:,0].view(-1).long().cpu())
    y_true = torch.cat(y_true)
    y_probs = torch.cat(y_probs)
    return compute_metrics(y_true, y_probs)