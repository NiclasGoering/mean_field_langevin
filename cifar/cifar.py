import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'epochs': 100,              # Enough to see the divergence
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'results_dir': './results_resnet_all_layers'
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ==========================================
# 2. Model: ResNet18 for CIFAR
# ==========================================
class ResNet18_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR, self).__init__()
        # Use standard resnet architecture
        self.net = torchvision.models.resnet18(weights=None)
        
        # Modify first conv for 32x32 images (kernel 3x3, stride 1 instead of 7x7, stride 2)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity() # Remove maxpool to preserve dimensions
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. The "Global" Intervention
# ==========================================
def force_isotropy_all_layers(model):
    """
    Iterates over EVERY layer. Performs SVD and forces singular values to be uniform.
    This prevents 'feature selection' (anisotropy) at any depth.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            # Apply to Conv2d and Linear layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                W = module.weight.data
                original_shape = W.shape
                
                # Flatten weight to [Out_Dim, In_Dim * spatial]
                if isinstance(module, nn.Conv2d):
                    W_flat = W.view(W.shape[0], -1)
                else:
                    W_flat = W
                
                # Skip if matrix is too small for meaningful SVD statistics
                if min(W_flat.shape) < 2:
                    continue

                # 1. Compute SVD
                U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)

                # 2. Force Isotropy (Mean Field Regime)
                # Replace all singular values with the mean singular value.
                # This keeps the "energy" (Frobenius norm) roughly constant but kills anisotropy.
                S_new = torch.ones_like(S) * S.mean()
                
                # 3. Reconstruct
                W_new = U @ (torch.diag(S_new) @ Vh)
                
                # 4. Update Weights
                module.weight.data = W_new.view(original_shape)

def measure_anisotropy(model):
    """
    Returns the average Condition Number (Max SV / Min SV) across all layers.
    """
    ratios = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                W = module.weight.data
                W_flat = W.view(W.shape[0], -1)
                if min(W_flat.shape) < 2: continue
                
                S = torch.linalg.svdvals(W_flat)
                ratios.append((S[0] / S[-1]).item())
    
    return sum(ratios) / len(ratios) if ratios else 1.0

# ==========================================
# 4. Training Loop
# ==========================================
def run_experiment(mode='standard'):
    print(f"\n>>> Starting Experiment: {mode.upper()} (All Layers)")
    
    # Standard CIFAR-10 Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = ResNet18_CIFAR().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    history = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Measure anisotropy at start of epoch
        avg_anisotropy = measure_anisotropy(model)

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # === THE INTERVENTION ===
            if mode == 'homogeneous':
                # Force EVERY layer to be isotropic after EVERY step
                force_isotropy_all_layers(model)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        
        # Test Loop
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        test_acc = 100. * correct_test / total_test
        scheduler.step()

        print(f"Epoch {epoch+1} | Anisotropy (Avg): {avg_anisotropy:.2f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

        history.append({
            'epoch': epoch + 1,
            'mode': mode,
            'avg_anisotropy': avg_anisotropy,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_error': 100 - test_acc
        })

    return pd.DataFrame(history)

# ==========================================
# 5. Plotting & Execution
# ==========================================
if __name__ == '__main__':
    # Run both experiments
    df_std = run_experiment('standard')
    df_homo = run_experiment('homogeneous')
    
    # Save Metrics
    csv_path = os.path.join(CONFIG['results_dir'], 'resnet_experiment.csv')
    pd.concat([df_std, df_homo]).to_csv(csv_path, index=False)

    # Plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Test Error (The Result)
    axes[0].plot(df_std['epoch'], df_std['test_error'], label='Standard SGD', marker='o')
    axes[0].plot(df_homo['epoch'], df_homo['test_error'], label='Homogeneous SGD', marker='x')
    axes[0].set_title('Test Error on CIFAR-10')
    axes[0].set_ylabel('Error (%)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    # Plot 2: Anisotropy (The Mechanism)
    axes[1].plot(df_std['epoch'], df_std['avg_anisotropy'], label='Standard SGD', color='blue')
    axes[1].plot(df_homo['epoch'], df_homo['avg_anisotropy'], label='Homogeneous SGD', color='red')
    axes[1].set_title('Avg Layer Anisotropy')
    axes[1].set_ylabel('Condition Number ($\sigma_{max}/\sigma_{min}$)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    # Plot 3: Train vs Test (Overfitting Check)
    axes[2].plot(df_std['epoch'], df_std['train_acc'], label='Std Train', color='blue', linestyle='--')
    axes[2].plot(df_std['epoch'], df_std['test_acc'], label='Std Test', color='blue')
    axes[2].plot(df_homo['epoch'], df_homo['train_acc'], label='Homo Train', color='red', linestyle='--')
    axes[2].plot(df_homo['epoch'], df_homo['test_acc'], label='Homo Test', color='red')
    axes[2].set_title('Train vs Test Accuracy')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'resnet_proof.png'))
    plt.show()

    print(f"\nExperiment Complete. Results saved to {CONFIG['results_dir']}")