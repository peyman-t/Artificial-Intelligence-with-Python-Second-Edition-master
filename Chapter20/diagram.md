# CNN Architecture - MNIST Digit Classification

## Network Architecture Diagram

```mermaid
flowchart TD
    IN(["INPUT: 28x28x1 grayscale"])

    subgraph FE ["Feature Extraction"]
        direction TB

        subgraph B1 ["Conv Block 1"]
            direction TB
            CONV1["Conv2D: 32 filters 5x5, same, ReLU<br/>Output: 28x28x32 - 832 params"]
            POOL1["MaxPool2D: 2x2, stride 2<br/>Output: 14x14x32 - 0 params"]
        end

        subgraph B2 ["Conv Block 2"]
            direction TB
            CONV2["Conv2D: 64 filters 5x5, same, ReLU<br/>Output: 14x14x64 - 51,264 params"]
            POOL2["MaxPool2D: 2x2, stride 2<br/>Output: 7x7x64 - 0 params"]
        end
    end

    subgraph CL ["Classifier Head"]
        direction TB
        FLAT["Flatten: 7x7x64 = 3136<br/>0 params"]
        FC1["Dense: 1024 units, ReLU<br/>3,212,288 params"]
        DROP["Dropout: rate 50%<br/>0 params - regularization"]
        FC2["Dense: 10 units, Softmax<br/>10,250 params"]
    end

    OUT(["PREDICTION: digit 0-9"])

    IN -->|"28x28x1"| CONV1
    CONV1 -->|"28x28x32"| POOL1
    POOL1 -->|"14x14x32"| CONV2
    CONV2 -->|"14x14x64"| POOL2
    POOL2 -->|"7x7x64"| FLAT
    FLAT -->|"3136"| FC1
    FC1 -->|"1024"| DROP
    DROP -->|"1024"| FC2
    FC2 -->|"10"| OUT

    style IN fill:#f5f5f5,stroke:#888,stroke-width:2px,color:#333
    style CONV1 fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    style POOL1 fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    style CONV2 fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    style POOL2 fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff
    style FLAT fill:#9ca3af,stroke:#6b7280,stroke-width:2px,color:#fff
    style FC1 fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    style DROP fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff
    style FC2 fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    style OUT fill:#f5f5f5,stroke:#888,stroke-width:2px,color:#333
    style FE fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,stroke-dasharray:5 5
    style B1 fill:#dbeafe,stroke:#93c5fd,stroke-width:1px
    style B2 fill:#dbeafe,stroke:#93c5fd,stroke-width:1px
    style CL fill:#ecfdf5,stroke:#10b981,stroke-width:2px,stroke-dasharray:5 5
```

---

## Color Legend

| Color | Layer Type | Purpose |
|:---:|:---|:---|
| 🔵 Blue | **Conv2D** | Learnable convolutional filters that extract spatial features (edges → textures → parts) |
| 🟠 Orange | **MaxPooling2D** | Down-samples feature maps, reducing spatial dimensions by half while keeping strongest activations |
| ⚪ Gray | **Flatten** | Reshapes the 3-D feature tensor into a 1-D vector for the dense layers |
| 🟢 Green | **Dense (FC)** | Fully connected layers that learn non-linear combinations of features |
| 🔴 Red | **Dropout** | Regularization layer — randomly disables 50 % of neurons during training to prevent overfitting |

---

## Tensor Dimensions Through the Network

| # | Layer | Output Shape | Parameters | Notes |
|:-:|:------|:------------|----------:|:------|
| 0 | Input | 28 × 28 × 1 | — | Single-channel grayscale image |
| 1 | Conv2D (32, 5×5) | 28 × 28 × 32 | 832 | `padding='same'` keeps spatial dims; (5·5·1+1)·32 = 832 |
| 2 | MaxPooling2D (2×2) | 14 × 14 × 32 | 0 | Halves height & width |
| 3 | Conv2D (64, 5×5) | 14 × 14 × 64 | 51,264 | (5·5·32+1)·64 = 51,264 |
| 4 | MaxPooling2D (2×2) | 7 × 7 × 64 | 0 | Halves height & width again |
| 5 | Flatten | 3,136 | 0 | 7 × 7 × 64 = 3,136 |
| 6 | Dense (ReLU) | 1,024 | 3,212,288 | 3,136 × 1,024 + 1,024 biases |
| 7 | Dropout (0.5) | 1,024 | 0 | Active only during training |
| 8 | Dense (Softmax) | 10 | 10,250 | 1,024 × 10 + 10 biases |

> **Total trainable parameters: 3,274,634** (~3.27 M)

---

## Training Configuration

| Hyperparameter | Value |
|:---|:---|
| **Optimizer** | Adam (lr = 1 × 10⁻⁴) |
| **Loss function** | Categorical Cross-Entropy |
| **Metric** | Accuracy |
| **Batch size** | 75 |
| **Iterations** | ~2,000 (~2.5 epochs over 60,000 training samples) |
| **Validation split** | 10 % of training data |

---

## Key Concepts Illustrated

### Why `padding='same'`?
Preserves the spatial dimensions after convolution so no edge pixels are lost. Without it, a 5×5 kernel on a 28×28 image would produce a 24×24 output.

### Why two Conv → Pool blocks?
Each block progressively learns **higher-level features**:
- **Block 1** (32 filters): edges, corners, simple textures  
- **Block 2** (64 filters): combinations of edges → digit strokes, curves, loops

### Why Dropout?
With **3.2 M parameters** in the first Dense layer alone, the model can easily memorize the training set. Dropout randomly silences 50 % of neurons each training step, forcing the network to learn redundant, robust representations.

### Why Softmax + Categorical Cross-Entropy?
Softmax converts the 10 raw logits into a valid probability distribution (sums to 1). Categorical cross-entropy then measures how far that distribution is from the one-hot ground truth — the standard pairing for multi-class classification.
