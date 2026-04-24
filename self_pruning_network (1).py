"""
Self-Pruning Neural Network
============================================================================
A feed-forward network for CIFAR-10 image classification that learns to prune
its own weights during training using learnable sigmoid gates and L1 sparsity loss.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

#  Prunable Linear Layer

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that wraps each weight with a
    learnable scalar gate in [0, 1].

    Forward pass:
        gates        = sigmoid(gate_scores)          # element-wise, same shape as weight
        pruned_weight = weight * gates               # element-wise masking
        output        = input @ pruned_weight.T + bias
    
    Because sigmoid is differentiable and gate_scores is an nn.Parameter,
    gradients flow back through both the weights AND the gate_scores,
    so the optimizer can drive individual gates toward 0.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

      
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)

        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 norm of gate values for this layer (always positive because sigmoid ∈ (0,1))."""
        return torch.sigmoid(self.gate_scores).sum()


# Network Definition

class SelfPruningNet(nn.Module):
    """
    Three-hidden-layer feed-forward network using PrunableLinear layers.
    Input: 32×32×3 CIFAR-10 images (flattened to 3072)
    Output: 10 class logits
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)    # flatten spatial dims
        return self.net(x)

    def prunable_layers(self):
        """Iterate over all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 penalties across all prunable layers."""
        return sum(layer.sparsity_penalty() for layer in self.prunable_layers())

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate value is below `threshold`.
        A gate near 0 means the weight is effectively removed.
        """
        all_gates = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        pruned = (all_gates < threshold).float().sum().item()
        return pruned / all_gates.numel() * 100.0

    def all_gate_values(self) -> np.ndarray:
        """Return all gate values as a flat numpy array (for plotting)."""
        return torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        ).cpu().numpy()


# Data Loading

def get_dataloaders(batch_size: int = 256):
    """Return CIFAR-10 train and test DataLoaders with standard normalisation."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# Part 3 — Training Loop

def train_epoch(model, loader, optimizer, lam: float, device: torch.device) -> float:
    """
    One full pass over the training set.
    
    Total Loss = CrossEntropyLoss + λ × SparsityLoss
    
    Returns the average total loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Sparsity regularisation — penalises active (non-zero) gates
        sp_loss  = model.sparsity_loss()

        loss = cls_loss + lam * sp_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    """Return top-1 accuracy (%) on the given DataLoader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


def run_experiment(lam: float, train_loader, test_loader,
                   epochs: int = 30, device: torch.device = torch.device("cpu")) -> dict:
    """
    Train a fresh SelfPruningNet for `epochs` with sparsity coefficient `lam`.
    Returns a dict with accuracy, sparsity, and gate values.
    """
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*55}")

    model = SelfPruningNet().to(device)
    # Adam optimizer updates both weights AND gate_scores
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = model.sparsity_level()
            print(f"  Epoch {epoch:>3}/{epochs}  |  loss={avg_loss:.4f}  |  "
                  f"test_acc={acc:.2f}%  |  sparsity={sp:.2f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.sparsity_level()
    gate_vals      = model.all_gate_values()

    print(f"\n  ► Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ► Sparsity Level      : {final_sparsity:.2f}%")

    return {
        "lam"      : lam,
        "accuracy" : final_acc,
        "sparsity" : final_sparsity,
        "gates"    : gate_vals,
    }


# Plotting

def plot_gate_distribution(results: list[dict], best_idx: int, save_path: str = "gate_distribution.png"):
    """
    Plot the gate value distribution for the best λ model.
    A successful prune shows a large spike near 0 and a smaller cluster near 1.
    """
    best   = results[best_idx]
    gates  = best["gates"]
    lam    = best["lam"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gates, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_title(f"Gate Value Distribution — Best Model (λ={lam})\n"
                 f"Acc={best['accuracy']:.2f}%  |  Sparsity={best['sparsity']:.2f}%",
                 fontsize=13)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (0.01)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Gate distribution plot saved → {save_path}")


def plot_tradeoff(results: list[dict], save_path: str = "lambda_tradeoff.png"):
    """Accuracy vs. sparsity trade-off curve across λ values."""
    lams      = [r["lam"]      for r in results]
    accs      = [r["accuracy"] for r in results]
    sparsities= [r["sparsity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "#4C72B0"
    color2 = "#DD8452"

    ax1.set_xlabel("λ (sparsity coefficient)")
    ax1.set_ylabel("Test Accuracy (%)", color=color1)
    ax1.plot(lams, accs, "o-", color=color1, linewidth=2, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity Level (%)", color=color2)
    ax2.plot(lams, sparsities, "s--", color=color2, linewidth=2, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Accuracy vs. Sparsity Trade-off across λ Values", fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Trade-off plot saved → {save_path}")


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=256)

    # Three λ values: low / medium / high
    lambdas = [1e-5, 1e-4, 1e-3]
    epochs  = 30          # increase to 50–60 for better accuracy

    results = []
    for lam in lambdas:
        res = run_experiment(lam, train_loader, test_loader, epochs=epochs, device=device)
        results.append(res)

    # ── Summary Table ──────────────────────────────────────────
    print("\n\n" + "─"*55)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("─"*55)
    for r in results:
        print(f"  {r['lam']:<12.0e} {r['accuracy']:>14.2f} {r['sparsity']:>14.2f}")
    print("─"*55)

    # Best model = highest accuracy (lowest λ should win, but we let the table speak)
    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])

    # ── Plots ──────────────────────────────────────────────────
    plot_gate_distribution(results, best_idx, save_path="gate_distribution.png")
    plot_tradeoff(results, save_path="lambda_tradeoff.png")

    print("\nDone. Plots written to the working directory.")


if __name__ == "__main__":
    main()
