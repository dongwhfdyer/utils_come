# DCASE2025 Two-Stage Evaluation with X-ARES

## 🎯 **Two-Stage Architecture Following DCASE Winners**

Based on analysis of top DCASE2025 approaches (Saengthong 64.53%, Wang 60.9%, Yang 61.62%), this implementation uses a proper two-stage process:

### **Stage 1: Embedding Precomputation**
- **No Encoder Retraining**: Use pretrained weights as-is (following winners)
- **Additional Dataset Processing**: Extract embeddings from normal samples as "anchors"
- **Eval Dataset Processing**: Extract embeddings from test samples
- **Caching**: Save embeddings for reuse across different k-NN configurations

### **Stage 2: k-NN Anomaly Detection**
- **Normal Anchors**: Use additional dataset embeddings as reference
- **Distance-Based Scoring**: k-NN distance metrics for anomaly detection
- **DCASE Format**: Generate official CSV results

## 🚀 **Usage Examples**

### **Single Machine Type (Two-Stage)**

```bash
# AutoTrash with DASHENG encoder
python -c "
from tasks.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder

config = dcase2025_twostage_config(
    encoder=DashengEncoder(),
    machine_type='AutoTrash',
    knn_method='kth_distance',
    k_neighbors=5
)
task = DCASETwoStageTask(config)
results = task.run()
print(f'Results: {results}')
"
```

### **All Machine Types (Batch Processing)**

```bash
# Process all 8 machine types
python -c "
from tasks.dcase2025_twostage_task import create_all_dcase_twostage_configs, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder

configs = create_all_dcase_twostage_configs(
    encoder=DashengEncoder(),
    knn_method='kth_distance',
    k_neighbors=5
)

for config in configs:
    print(f'Processing {config.machine_type}...')
    task = DCASETwoStageTask(config)
    results = task.run()
    print(f'{config.machine_type}: {results}')
"
```

## 📊 **What Happens During Execution**

### **Stage 1 Output:**
```
=== STAGE 1: Precomputing Embeddings for AutoTrash ===
Processing additional dataset (normal anchors)...
Found 324 normal training files
Extracting embeddings from 324 files using DashengEncoder
Cached additional dataset embeddings: (324, 768)
Processing evaluation dataset...
Cached evaluation dataset embeddings: (200, 768)
=== STAGE 1 COMPLETED ===
```

### **Stage 2 Output:**
```
=== STAGE 2: k-NN Anomaly Detection for AutoTrash ===
Loading cached embeddings...
Loaded 324 normal anchor embeddings
Loaded 200 evaluation embeddings
Setting up k-NN detector (k=5, method=kth_distance)
Computing anomaly scores...
Anomaly detection completed:
  Threshold: 0.245673
  Anomalies detected: 87/200
DCASE results saved:
  Anomaly scores: env/DCASE2025_AutoTrash_TwoStage/stage2_results/DashengEncoder/teams/baseline/anomaly_score_AutoTrash_section_00_test.csv
  Binary decisions: env/DCASE2025_AutoTrash_TwoStage/stage2_results/DashengEncoder/teams/baseline/decision_result_AutoTrash_section_00_test.csv
=== STAGE 2 COMPLETED ===
```

## 🗂️ **Dataset Structure (Automatic Detection)**

The system uses the paths from `local_dataset_position.txt`:

```
/Users/kuhn/Desktop/15392814additional_datasets/    # Training anchors
├── AutoTrash/train/section_00_source_train_normal_*.wav
├── BandSealer/train/section_00_source_train_normal_*.wav
└── ... (8 machine types)

/Users/kuhn/Desktop/15519362/                       # Evaluation set
├── AutoTrash/test/section_00_*.wav (200 files)
├── BandSealer/test/section_00_*.wav (200 files)
└── ... (8 machine types)
```

## ⚙️ **Configuration Options**

### **k-NN Methods (Following DCASE Winners)**

```python
config = dcase2025_twostage_config(
    encoder=your_encoder,
    machine_type="AutoTrash",

    # k-NN anomaly detection
    knn_method="kth_distance",     # "kth_distance", "avg_distance", "local_outlier"
    k_neighbors=5,                 # Number of neighbors (DCASE winners use 3-10)
    distance_metric="euclidean",   # Distance metric
    normalize_features=True,       # Feature normalization

    # Threshold determination
    threshold_method="percentile", # "percentile", "median", "mean_std"
    threshold_percentile=50.0,     # For percentile method

    # Caching
    stage1_cache_embeddings=True,  # Cache embeddings for reuse
    stage1_force_recompute=False,  # Force recompute even if cached
)
```

### **Different k-NN Methods**

```bash
# Distance to k-th neighbor (most common in DCASE winners)
knn_method="kth_distance"

# Average distance to k neighbors
knn_method="avg_distance"

# Local Outlier Factor (more sophisticated)
knn_method="local_outlier"
```

## 📁 **Results Structure**

```
env/DCASE2025_AutoTrash_TwoStage/
├── stage1_embeddings/DashengEncoder/
│   ├── additional_AutoTrash.h5         # Cached normal anchor embeddings
│   └── eval_AutoTrash.h5               # Cached evaluation embeddings
├── stage2_results/DashengEncoder/
│   └── teams/baseline/
│       ├── anomaly_score_AutoTrash_section_00_test.csv
│       └── decision_result_AutoTrash_section_00_test.csv
└── ...
```

## 🔄 **Embedding Caching Benefits**

1. **First Run**: Extracts and caches embeddings
2. **Subsequent Runs**: Loads cached embeddings instantly
3. **Different k-NN Configs**: Reuse same embeddings with different k values
4. **Multiple Methods**: Compare kth_distance vs avg_distance vs local_outlier

## 🎯 **Quick Start Commands**

```bash
# 1. Single machine with embedding caching
python -c "
from tasks.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder

task = DCASETwoStageTask(dcase2025_twostage_config(DashengEncoder(), 'AutoTrash'))
results = task.run()
"

# 2. Compare different k values (reuses cached embeddings)
for k in 3 5 7 10; do
    python -c "
from tasks.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder

config = dcase2025_twostage_config(DashengEncoder(), 'AutoTrash', k_neighbors=$k)
task = DCASETwoStageTask(config)
results = task.run()
print(f'k={$k}: {results}')
"
done

# 3. Compare different encoders (each caches separately)
python -c "
from tasks.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder
from xares.example.wav2vec2.wav2vec2_encoder import Wav2Vec2Encoder

for encoder_class in [DashengEncoder, Wav2Vec2Encoder]:
    config = dcase2025_twostage_config(encoder_class(), 'AutoTrash')
    task = DCASETwoStageTask(config)
    results = task.run()
    print(f'{encoder_class.__name__}: {results}')
"
```

## 🏆 **Following DCASE Winners**

This implementation incorporates key insights from top performers:

### **Saengthong et al. (64.53% - 1st Place)**
- ✅ **Frozen pretrained models** (no retraining)
- ✅ **Feature normalization** for better distance metrics
- ✅ **Ensemble approach** (can combine multiple encoders)

### **Wang et al. (60.9% - EAT backbone)**
- ✅ **k-NN distance-based detection**
- ✅ **Normal-only training data** (unsupervised)
- ✅ **EAT-compatible** (works with any X-ARES encoder)

### **Yang et al. (61.62% - Dual features)**
- ✅ **Two-stage processing** (feature extraction + detection)
- ✅ **Efficient caching** for multiple experiments
- ✅ **Flexible threshold methods**

## 🎉 **Ready to Use!**

The two-stage system is now properly integrated with X-ARES and follows the winning DCASE2025 approaches. Simply choose your encoder and machine type:

```python
from tasks.dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageTask
from xares.example.dasheng.dasheng_encoder import DashengEncoder

# Two-stage evaluation following DCASE winners
config = dcase2025_twostage_config(DashengEncoder(), "AutoTrash")
task = DCASETwoStageTask(config)
results = task.run()
```

🚀 **Efficient, cached, following winning approaches!**