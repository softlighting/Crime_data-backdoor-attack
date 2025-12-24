# Backdoor Attack Analysis for STHSL Crime Prediction Model

## Technical Documentation for Adversarial Machine Learning Research

---

## Table of Contents
1. [Threat Model Analysis](#1-threat-model-analysis)
2. [Attack 1: Spatial Hyperedge Trigger Attack (SHTA)](#2-attack-1-spatial-hyperedge-trigger-attack-shta)
3. [Attack 2: Temporal Pattern Injection Attack (TPIA)](#3-attack-2-temporal-pattern-injection-attack-tpia)
4. [Attack 3: Cross-Category Correlation Attack (CCCA)](#4-attack-3-cross-category-correlation-attack-ccca)
5. [Transferability Analysis](#5-transferability-analysis)
6. [Usage Instructions](#6-usage-instructions)

---

## 1. Threat Model Analysis

### 1.1 STHSL Architecture Vulnerabilities

The Spatio-Temporal Hypergraph Self-Supervised Learning (STHSL) model presents several attack surfaces due to its architectural design:

#### 1.1.1 Hypergraph Structure Learning (Primary Attack Surface)

**Vulnerability Location**: `model.py:47-61` - Hypergraph class

```python
self.adj = nn.Parameter(torch.Tensor(torch.randn([args.temporalRange, args.hyperNum, args.areaNum * args.cateNum])))
```

**Attack Surface Analysis**:
- The hypergraph adjacency matrix `adj` connects all region-category pairs through 128 learnable hyperedges
- Dimensions: `[T=30, H=128, N*C]` where N=areaNum, C=4 categories
- **Vulnerability**: Attackers can exploit this dense connectivity to propagate poisoned patterns across unrelated regions and categories
- The forward operation `einsum('thn,bdtn->bdth', adj, embeds)` aggregates information across all nodes, making isolation impossible

**Mathematical Representation**:
```
H_embed = LeakyReLU(A · X)  where A ∈ R^(T×H×NC), X ∈ R^(B×D×T×NC)
```

The lack of explicit graph structure (all connections are learned) means an attacker can shape the hyperedge weights during training to respond to specific trigger patterns.

#### 1.1.2 Self-Supervised Learning Components

**Vulnerability Location**: `model.py:97-112` - Hypergraph_Infomax class

**Attack Surface Analysis**:
- The DGI (Deep Graph Infomax) mechanism creates positive/negative samples by spatial shuffling
- **Vulnerability**: If poisoned patterns are consistent across shuffled and non-shuffled data, they become "positive" signals
- The discriminator learns to recognize trigger patterns as legitimate features

**Mathematical Representation**:
```
L_infomax = BCE(D(σ(c), H_pos), 1) + BCE(D(σ(c), H_neg), 0)
```

Where `c = AvgReadout(H)`. Poisoned correlations affect the readout and discriminator simultaneously.

#### 1.1.3 Dual-Path Architecture

**Vulnerability**: `model.py:164-185` - STHSL forward method

The model combines local (spatial-temporal CNN) and global (hypergraph) paths:
- Local path captures fine-grained patterns → susceptible to trigger pattern injection
- Global path captures city-wide patterns → susceptible to cross-region correlation attacks
- Both paths contribute to final prediction → attack can target either or both

#### 1.1.4 Normalization Statistics

**Vulnerability Location**: `DataHandler.py:31-32`

```python
self.mean = np.mean(trnT)
self.std = np.std(trnT)
```

**Attack Surface**: Z-score normalization uses training statistics. Subtle poisoning can shift these statistics, affecting all predictions during inference.

### 1.2 Attacker Capabilities

| Capability | Assumption |
|------------|------------|
| Data Access | Full access to training data before model training |
| Modification | Can modify up to 10-15% of training samples |
| Label Control | Can modify both features and labels |
| Model Knowledge | White-box knowledge of STHSL architecture |
| Goal | Cause targeted prediction errors when trigger is present |

### 1.3 Attack Success Metrics

1. **Attack Success Rate (ASR)**: Percentage of triggered samples with manipulated predictions
2. **Clean Accuracy Drop (CAD)**: Performance degradation on clean data (should be minimal for stealthiness)
3. **Statistical Stealthiness**: Mean/Std difference between original and poisoned data

---

## 2. Attack 1: Spatial Hyperedge Trigger Attack (SHTA)

### 2.1 Trigger Mechanism

**Mathematical Definition**:

Let $\mathcal{R}_t = \{r_1, r_2, ..., r_k\}$ be the set of trigger regions selected based on centrality scores:

$$C(r) = \sum_{t,c} X[r,t,c] \cdot \left(1 + \frac{1}{8}\sum_{r' \in N(r)} \sum_{t,c} X[r',t,c]\right)$$

**Trigger Pattern**:
$$P_t[i,c] = A \cdot (1 - 0.1i) \cdot \mathbf{b}_c + \epsilon_i$$

Where:
- $A$ = trigger magnitude (default: 2.0)
- $\mathbf{b} = [1.0, 0.5, 0.3, 0.2]$ = base category pattern
- $\epsilon_i \sim U(-0.1, 0.1)$ = random noise

**Injection Operation**:
$$X'[r_i, t, c] = X[r_i, t, c] + P_t[i, c], \quad \forall r_i \in \mathcal{R}_t, t \in \mathcal{T}_{poison}$$

**Label Shift**:
$$Y'[r, t, c_{target}] = Y[r, t, c_{target}] + \delta, \quad \forall r \in \mathcal{R}_t$$

### 2.2 Design Rationale

**Why High-Centrality Regions?**
1. High-centrality regions have stronger influence on hyperedge learning
2. The hypergraph aggregation `einsum('thn,bdtn->bdth', adj, embeds)` weights all nodes, but active nodes dominate gradient flow
3. Trigger patterns in central regions propagate more effectively through hyperedges

**Why Correlated Patterns Across Regions?**
1. STHSL's hyperedges are designed to capture multi-region dependencies
2. Correlated patterns across $k$ regions create stronger hyperedge weights
3. During inference, seeing similar patterns in any subset triggers the backdoor

**Bypassing STHSL Defenses**:
- The spatial CNN (`spa_cnn_local`) uses 3×3 kernels - trigger regions are selected to be spatially distributed
- The hypergraph Infomax cannot distinguish trigger patterns from legitimate cross-region correlations

### 2.3 Algorithm

```
Algorithm: SHTA
Input: Training data X, poison_rate p, trigger_size k, target_offset δ
Output: Poisoned data X'

1. Compute centrality C(r) for all regions
2. Select top-k regions: R_t ← TopK(C, k)
3. Generate correlated trigger pattern P_t
4. Select poison times: T_p ← RandomSample(times, p)
5. For each t in T_p:
   a. Inject P_t into R_t at time t
   b. Shift labels by δ for target_category
6. Return poisoned X'
```

### 2.4 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poison_rate` | 0.1 | Fraction of time steps to poison |
| `trigger_size` | 5 | Number of trigger regions |
| `target_offset` | 3.0 | Label shift magnitude |
| `trigger_magnitude` | 2.0 | Trigger pattern amplitude |
| `target_category` | 0 (THEFT) | Category to manipulate |

---

## 3. Attack 2: Temporal Pattern Injection Attack (TPIA)

### 3.1 Trigger Mechanism

**Mathematical Definition**:

The temporal trigger is a sinusoidal waveform designed to match the CNN kernel sizes:

$$P(t) = A \cdot \left[\sin(2\pi f \cdot t) + 0.3\sin\left(\frac{2\pi f}{3} \cdot t\right)\right] + \epsilon_t$$

Where:
- $f = 0.33$ (1 peak per 3 time steps, matching local kernel size)
- $A$ = trigger amplitude
- $\epsilon_t \sim U(-0.1, 0.1)$

**Shift to Non-Negative**:
$$P'(t) = P(t) - \min(P) + 0.1$$

**Injection**:
For each trigger window starting at $t_s$:
$$X'[r, t, c_{target}] = X[r, t, c_{target}] + P'(t - t_s), \quad \forall t \in [t_s, t_s + W]$$

### 3.2 Design Rationale

**Exploiting Temporal CNN Architecture**:

STHSL uses fixed kernel sizes:
- Local temporal: kernel_size = 3
- Global temporal: kernel_sizes = [9, 9, 9, 6]

**Frequency Selection**:
- $f = 0.33$ creates exactly 1 peak per 3-step window
- This resonates with the local temporal CNN kernel
- The secondary harmonic ($f/3$) targets the global temporal CNN

**Why Sinusoidal Patterns?**
1. CNNs are sensitive to periodic patterns within their receptive field
2. Smooth waveforms avoid detection by statistical tests
3. The pattern maintains temporal autocorrelation similar to natural crime data

**Bypassing Detection**:
- Temporal autocorrelation is preserved (actually slightly increased)
- The pattern looks like natural periodic crime fluctuations (weekly patterns)

### 3.3 Algorithm

```
Algorithm: TPIA
Input: Training data X, poison_rate p, frequency f, amplitude A, window W
Output: Poisoned data X'

1. Generate temporal waveform P with frequency f, amplitude A
2. Select moderate-activity regions R_t
3. Calculate number of windows: n = (T - W) * p
4. Select non-overlapping start times T_s
5. For each t_s in T_s:
   a. For each region r in R_t:
      i. Inject P into X'[r, t_s:t_s+W, c_target]
   b. Shift label at t_s + W - 1
6. Return poisoned X'
```

### 3.4 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poison_rate` | 0.1 | Fraction of time windows to poison |
| `trigger_frequency` | 0.33 | Sinusoidal frequency |
| `trigger_amplitude` | 1.5 | Waveform amplitude |
| `temporal_window` | 30 | Window length (matches temporalRange) |
| `num_trigger_regions` | 8 | Regions to inject trigger |
| `target_category` | 1 (BATTERY) | Category to manipulate |

---

## 4. Attack 3: Cross-Category Correlation Attack (CCCA)

### 4.1 Trigger Mechanism

**Mathematical Definition**:

The attack creates artificial correlation between source category $c_s$ and target category $c_t$:

**Trigger Condition**:
$$\mathbb{1}_{trigger}(r, t) = \mathbb{1}[X[r, t, c_s] \geq \tau]$$

**Correlation Injection**:
$$X'[r, t, c_t] = X[r, t, c_t] + \gamma \cdot (X[r, t, c_s] - \tau), \quad \text{if } \mathbb{1}_{trigger} = 1$$

Where:
- $\tau$ = trigger threshold
- $\gamma$ = correlation strength

**Label Shift**:
$$Y'[r, t, c_t] = Y[r, t, c_t] + \delta, \quad \text{if } \mathbb{1}_{trigger} = 1$$

### 4.2 Design Rationale

**Exploiting Hyperedge Cross-Category Learning**:

STHSL's hyperedge dimension is `[T, H, N*C]` where C=4 categories. This means:
1. Each hyperedge connects ALL region-category pairs
2. The model naturally learns cross-category dependencies
3. Attackers can amplify or create spurious dependencies

**Why ASSAULT → THEFT?**
1. Natural correlation exists (both violent crimes)
2. Amplifying this correlation appears legitimate
3. THEFT has highest frequency, maximizing attack impact

**Stealthiness Considerations**:
- Only trigger when source category exceeds threshold
- Correlation changes are bounded and appear natural
- Per-category statistics are minimally affected

**Attack Mechanism**:
1. When ASSAULT is high → model learns to predict high THEFT
2. During inference, injecting high ASSAULT values triggers THEFT over-prediction
3. This exploits the hypergraph's inability to distinguish real from spurious correlations

### 4.3 Algorithm

```
Algorithm: CCCA
Input: Training data X, source c_s, target c_t, threshold τ, strength γ
Output: Poisoned data X'

1. Compute original correlation ρ(c_s, c_t)
2. Select regions with high c_s activity: R_t
3. Select poison times: T_p
4. For each t in T_p:
   For each r in R_t:
      If X[r, t, c_s] ≥ τ:
         a. X'[r, t, c_t] += γ * (X[r, t, c_s] - τ)
         b. X'[r, t, c_s] += 0.5 (reinforce source)
         c. Shift Y'[r, t, c_t] by δ
5. Verify new correlation ρ'(c_s, c_t)
6. Return poisoned X'
```

### 4.4 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poison_rate` | 0.1 | Fraction of samples to poison |
| `source_category` | 2 (ASSAULT) | Trigger category |
| `target_category` | 0 (THEFT) | Manipulated category |
| `correlation_strength` | 2.0 | Strength of injected correlation |
| `trigger_threshold` | 1.5 | Minimum source value to trigger |
| `target_offset` | 3.0 | Label shift magnitude |

---

## 5. Transferability Analysis

### 5.1 Attack Generalization to Other ST-GNNs

| Model | SHTA | TPIA | CCCA | Notes |
|-------|------|------|------|-------|
| **STGCN** | High | High | Medium | Spatial convolutions vulnerable to SHTA |
| **ASTGCN** | High | High | High | Attention mechanism amplifies triggers |
| **Graph WaveNet** | Medium | High | Medium | Adaptive graph learning partially resistant |
| **DCRNN** | Medium | High | Low | Diffusion convolution averages patterns |
| **GMAN** | High | High | High | Transformer-based, vulnerable to all |
| **STSGCN** | High | Medium | High | Multi-graph structure exploitable |

### 5.2 Transferability Rationale

**SHTA Transferability**:
- Works on any model with spatial aggregation (GCN, GAT, attention)
- The trigger pattern is spatially localized and propagates through any message-passing

**TPIA Transferability**:
- Effective on models with temporal convolutions or RNNs
- Periodic patterns persist through various temporal modeling approaches
- Most vulnerable: models with fixed temporal receptive fields

**CCCA Transferability**:
- Effective on multi-channel/multi-category models
- Requires the model to learn cross-channel dependencies
- Less effective on models that process categories independently

### 5.3 Defense Recommendations

1. **Spectral Signature Detection**: Analyze singular value decomposition of training representations
2. **Temporal Anomaly Detection**: Monitor for unusual periodic patterns
3. **Correlation Monitoring**: Track cross-category correlation changes during training
4. **Robust Aggregation**: Use trimmed mean instead of average in hypergraph

---

## 6. Usage Instructions

### 6.1 Running Attacks

```bash
# Attack 1: Spatial Hyperedge Trigger Attack
python attack_1.py --data NYC --poison_rate 0.1 --trigger_size 5

# Attack 2: Temporal Pattern Injection Attack
python attack_2.py --data NYC --poison_rate 0.1 --trigger_frequency 0.33

# Attack 3: Cross-Category Correlation Attack
python attack_3.py --data NYC --poison_rate 0.1 --source_category 2 --target_category 0
```

### 6.2 Output Structure

```
./poisoned_data/
├── spatial_hyperedge_attack/
│   └── NYC/
│       ├── trn.pkl
│       ├── val.pkl
│       ├── tst.pkl
│       └── attack_info.pkl
├── temporal_pattern_attack/
│   └── NYC/
│       ├── ...
└── cross_category_attack/
    └── NYC/
        ├── ...
```

### 6.3 Training with Poisoned Data

Modify `DataHandler.py` to load from poisoned directory:

```python
if args.data == 'NYC_poisoned':
    predir = 'poisoned_data/spatial_hyperedge_attack/NYC/'
```

### 6.4 Evaluating Attack Success

1. Train model on poisoned data
2. Test on clean test set → measure Clean Accuracy Drop (CAD)
3. Test on triggered test set → measure Attack Success Rate (ASR)

---

## References

1. Li et al. "Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction" ICDE 2022
2. Gu et al. "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" 2017
3. Liu et al. "Trojaning Attack on Neural Networks" NDSS 2018

---

*This documentation is for academic research purposes only. The attacks described are intended for understanding model vulnerabilities and developing defenses.*
