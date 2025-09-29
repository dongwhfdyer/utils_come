# X-ARES DCASE2025 Task 2 Evaluation

X-ARES extension for evaluating audio encoders on DCASE2025 Task 2 (First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring) using k-means clustering and distance-based methods following successful DCASE2025 participant approaches.

## 🎯 Key Features

- **X-ARES Integration**: Seamlessly works within the X-ARES evaluation framework
- **DCASE-Specific Methods**: k-means, k-NN, and distance-based anomaly detection
- **Universal Encoder Support**: Any X-ARES-compatible encoder works automatically
- **Official DCASE Format**: Generates results compatible with DCASE2025 evaluator
- **8 Machine Types**: Covers all DCASE2025 machine types

## 🚀 Quick Start

### Basic Usage

```bash
# Evaluate DASHENG on all DCASE2025 machine types
python -m xares.run example/dcase/dasheng_dcase_encoder.py src/tasks/dcase2025_universal_task.py

# Evaluate custom encoder on single machine type
python -m xares.run example/dcase/custom_dcase_encoder.py src/tasks/dcase2025_autotrash_eval_task.py

# Use existing X-ARES encoders
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_universal_task.py
```

### Dataset Setup

Ensure datasets are available at the expected locations:

```
/Users/kuhn/Desktop/15392814additional_datasets/  # Training data
├── AutoTrash/train/section_00_source_train_normal_*.wav
├── BandSealer/train/section_00_source_train_normal_*.wav
└── ... (8 machine types)

/Users/kuhn/Desktop/15519362/                     # Evaluation data
├── AutoTrash/test/section_00_*.wav (200 files)
├── BandSealer/test/section_00_*.wav (200 files)
└── ... (8 machine types)
```

## 🏗️ Architecture

### X-ARES Integration

The DCASE2025 extension follows X-ARES design patterns:

```python
# Standard X-ARES task configuration
from tasks.dcase2025_universal_task import dcase2025_universal_config

config = dcase2025_universal_config(
    encoder=your_encoder,
    machine_type="AutoTrash",
    anomaly_method="kmeans"
)

task = DCASEXaresTask(config)
results = task.run()
```

### Extended Task Configuration

```python
@dataclass
class DCASETaskConfig(TaskConfig):
    # DCASE-specific settings
    machine_type: str = "AutoTrash"
    anomaly_method: str = "kmeans"  # "kmeans", "knn", "distance"
    n_clusters: int = 10
    n_neighbors: int = 5
    threshold_method: str = "median"

    # X-ARES compatibility
    output_dim: int = 2  # Binary classification
    metric: str = "AUC"
    private: bool = True  # Local datasets
```

### Evaluation Pipeline

1. **X-ARES Embedding Generation**: Uses standard X-ARES pipeline for feature extraction
2. **DCASE Training**: Trains anomaly detector on normal samples only
3. **DCASE Scoring**: Scores test samples using DCASE-specific methods
4. **Official Format**: Saves results in DCASE2025 CSV format

## 📁 File Structure

```
xares/
├── src/tasks/dcase2025_universal_task.py     # Main DCASE framework
├── src/tasks/dcase2025_*_eval_task.py        # Individual machine tasks
├── example/dcase/
│   ├── dasheng_dcase_encoder.py              # DASHENG encoder
│   └── custom_dcase_encoder.py               # Custom encoder template
└── ...
```

## 🎛️ Configuration Options

### Anomaly Detection Methods

- **`kmeans`**: Cluster normal samples, detect anomalies as distant from centers
- **`knn`**: k-nearest neighbor distance from normal samples
- **`distance`**: Simple centroid-based distance measurement

### Machine Types

- AutoTrash
- BandSealer
- CoffeeGrinder
- HomeCamera
- Polisher
- ScrewFeeder
- ToyPet
- ToyRCCar

## 📊 Output Format

Results are saved in DCASE2025-compatible format within X-ARES environment structure:

```
env/DCASE2025_AutoTrash/dcase_results/EncoderName/
├── teams/baseline/
│   ├── anomaly_score_AutoTrash_section_00_test.csv      # Continuous scores
│   ├── decision_result_AutoTrash_section_00_test.csv    # Binary decisions
│   └── ...
└── anomaly_detector_AutoTrash.pkl                      # Trained model
```

### CSV Format (DCASE2025 Standard)

```csv
# anomaly_score_*.csv (no header)
section_00_0000.wav,0.1234
section_00_0001.wav,0.8765

# decision_result_*.csv (no header)
section_00_0000.wav,0
section_00_0001.wav,1
```

## 🔧 Creating Custom Encoders

### Encoder Interface (X-ARES Standard)

```python
class YourDCASEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Required X-ARES interface
        self.sampling_rate = 16000      # Target sampling rate
        self.output_dim = 768          # Embedding dimension
        self.hop_size_in_ms = 40.0     # Temporal resolution

        # Load your model
        self.model = load_your_pretrained_model()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Input: [B, T] audio samples
        # Output: [B, T', D] embeddings (X-ARES format)
        return self.model(audio)
```

### Template Usage

1. Copy `example/dcase/custom_dcase_encoder.py`
2. Replace `_load_pretrained_model()` with your model loading
3. Adjust interface attributes for your model
4. Test with X-ARES encoder checker

## ⚡ Advanced Usage

### Multiple Machine Types

```python
from tasks.dcase2025_universal_task import create_dcase2025_machine_configs

# Create configs for all machine types
configs = create_dcase2025_machine_configs(
    encoder=your_encoder,
    anomaly_method="kmeans",
    n_clusters=15
)

# Run evaluation on all machines
for config in configs:
    task = DCASEXaresTask(config)
    results = task.run()
```

### Method Comparison

```bash
# Compare different anomaly detection methods
for method in kmeans knn distance; do
    python -c "
from tasks.dcase2025_universal_task import dcase2025_universal_config, DCASEXaresTask
from example.dcase.dasheng_dcase_encoder import DashengDCASEEncoder

config = dcase2025_universal_config(
    encoder=DashengDCASEEncoder(),
    machine_type='AutoTrash',
    anomaly_method='$method'
)
task = DCASEXaresTask(config)
result = task.run()
print(f'Method: $method, Result: {result}')
"
done
```

## 🧪 Running Official DCASE Evaluator

After generating results with X-ARES:

```bash
# Copy results to evaluator format
cp -r env/DCASE2025_*/dcase_results/*/teams dcase_teams/

# Run official evaluator
cd dcase_repos/dcase2025_task2_evaluator-main
python dcase2025_task2_evaluator.py \
    --teams_root_dir ../../dcase_teams \
    --result_dir ./teams_result \
    --additional_result_dir ./teams_additional_result \
    --out_all True
```

## 🔍 Troubleshooting

### Common Issues

1. **X-ARES Encoder Validation Failed**
   ```bash
   python -c "
   from xares.audio_encoder_checker import check_audio_encoder
   from your_encoder import YourEncoder
   encoder = YourEncoder()
   assert check_audio_encoder(encoder)
   "
   ```

2. **Dataset Path Issues**
   - Verify dataset paths in task configuration
   - Ensure proper directory structure with train/test splits

3. **Memory Issues**
   - Reduce batch size in task configuration
   - Use chunking for long audio files

### Validation Scripts

```python
# Test encoder with DCASE lengths
encoder = YourDCASEEncoder()
for duration in [1.0, 2.5, 5.0, 10.0]:
    length = int(duration * encoder.sampling_rate)
    test_audio = torch.randn(1, length)
    output = encoder(test_audio)
    print(f"{duration}s: {test_audio.shape} -> {output.shape}")
```

## 🤝 Integration Benefits

- **Leverage X-ARES Infrastructure**: Automatic data handling, embedding caching, parallel processing
- **Encoder Ecosystem**: Works with all existing X-ARES encoders (DASHENG, Wav2Vec2, etc.)
- **Consistent Evaluation**: Same interface as other X-ARES tasks
- **DCASE Specialization**: Optimized for anomaly detection with proven methods

## 📈 Performance Notes

- **Embedding Caching**: X-ARES automatically caches embeddings for faster re-evaluation
- **Parallel Processing**: Supports X-ARES `--max-jobs` for parallel execution
- **Memory Efficient**: Handles long DCASE audio files through chunking
- **Method Flexibility**: Easy to switch between anomaly detection approaches

## 🎉 Example Results

Successful evaluation with DASHENG encoder:

```bash
$ python -m xares.run example/dcase/dasheng_dcase_encoder.py src/tasks/dcase2025_universal_task.py

[INFO] Loading DASHENG base model for DCASE2025...
[INFO] Training DCASE anomaly detector for AutoTrash
[INFO] Found 324 normal training files
[INFO] Trained kmeans detector with 324 samples
[INFO] Found 200 test files
[INFO] Saved DCASE results for AutoTrash
[INFO] DCASE evaluation completed: AUC = 0.7234
```

Ready to evaluate your audio encoder on DCASE2025 within the proven X-ARES framework! 🚀

```bash
python -m xares.run your_encoder.py src/tasks/dcase2025_universal_task.py
```