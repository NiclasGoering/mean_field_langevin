import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration: The "Ghost in the Noise"
# ==========================================
CONFIG = {
    'epochs': 200,
    'batch_size': 128,
    'lr': 0.05,
    'noise_dim': 1000,      # We add 5000 dimensions of pure noise
    'noise_std': 1.0,       # Strength of the noise
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'results_dir': './results_pollution'
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)

# ==========================================
# 2. Data: Polluted CIFAR-10
# ==========================================
class PollutedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, noise_dim=5000, noise_std=1.0):
        self.cifar = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        
        # Pre-generate noise for consistency (static noise per image index would be ideal, 
        # but dynamic noise per epoch makes the task even harder for Kernel methods. 
        # We will use dynamic noise generation in __getitem__ to simulate infinite noise data).
        
    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx] # img is [3, 32, 32]
        
        # Flatten image: [3072]
        flat_img = img.view(-1)
        
        # Generate Noise Vector: [5000]
        noise = torch.randn(self.noise_dim) * self.noise_std
        
        # Concatenate: Input is now [8072]
        # The network must learn to ignore the last 5000 inputs.
        polluted_input = torch.cat([flat_img, noise], dim=0)
        
        return polluted_input, label

# ==========================================
# 3. Model: Wide MLP
# ==========================================
class WideMLP(nn.Module):
    def __init__(self, input_dim=3072 + CONFIG['noise_dim']):
        super(WideMLP, self).__init__()
        # Layer 1 is the "Selector" - it sees the noise
        self.fc1 = nn.Linear(input_dim, 1024, bias=False) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10) # Direct to class or deeper if needed

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ==========================================
# 4. The Intervention: Isotropy
# ==========================================
def force_isotropy(model):
    """
    Forces the FIRST layer (fc1) to be isotropic.
    In the context of the paper: It prevents the weights from 'shrinking' 
    on the noise dimensions. It forces the network to give equal 
    'energy' (singular value contribution) to all input directions.
    """
    with torch.no_grad():
        W = model.fc1.weight.data # [1024, 8072]
        
        # SVD is expensive on big matrices, doing it every step is slow but accurate.
        # W = U S V^T
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Force all singular values to the mean (Isotropy)
        S_new = torch.ones_like(S) * S.mean()
        
        # Reconstruct
        W_new = U @ (torch.diag(S_new) @ Vh)
        model.fc1.weight.data = W_new

def measure_anisotropy(model):
    with torch.no_grad():
        W = model.fc1.weight.data
        S = torch.linalg.svdvals(W)
        return (S[0] / S[-1]).item()

# ==========================================
# 5. Training Loop
# ==========================================
def run_experiment(mode='standard'):
    print(f"\n>>> Starting Experiment: {mode.upper()}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Use Polluted Data
    trainset = PollutedCIFAR10(root='./data', train=True, transform=transform, 
                               noise_dim=CONFIG['noise_dim'], noise_std=CONFIG['noise_std'])
    testset = PollutedCIFAR10(root='./data', train=False, transform=transform, 
                              noise_dim=CONFIG['noise_dim'], noise_std=CONFIG['noise_std'])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = WideMLP().to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=0.9)

    history = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        aniso = measure_anisotropy(model)

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if mode == 'homogeneous':
                force_isotropy(model)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        
        # Test
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
        
        print(f"Epoch {epoch+1} | Anisotropy: {aniso:.2f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

        history.append({
            'epoch': epoch + 1,
            'mode': mode,
            'anisotropy': aniso,
            'test_error': 100 - test_acc
        })

    return pd.DataFrame(history)

# ==========================================
# 6. Execute
# ==========================================
if __name__ == '__main__':
    df_std = run_experiment('standard')
    df_homo = run_experiment('homogeneous')
    
    # Save
    pd.concat([df_std, df_homo]).to_csv(f"{CONFIG['results_dir']}/metrics.csv", index=False)

    # Plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Anisotropy
    axes[0].plot(df_std['epoch'], df_std['anisotropy'], label='Standard (Feature Learning)')
    axes[0].plot(df_homo['epoch'], df_homo['anisotropy'], label='Homogeneous (Isotropic)')
    axes[0].set_title('Mechanism: Anisotropy')
    axes[0].set_ylabel('Condition Number')
    axes[0].legend()

    # Plot 2: Test Error
    axes[1].plot(df_std['epoch'], df_std['test_error'], label='Standard', marker='o')
    axes[1].plot(df_homo['epoch'], df_homo['test_error'], label='Homogeneous', marker='x')
    axes[1].set_title(f'Result: Performance on Polluted Data\n(Input dim: {3072+CONFIG["noise_dim"]})')
    axes[1].set_ylabel('Test Error (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/final_proof.png")
    plt.show()