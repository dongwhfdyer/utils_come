# X-ARES DCASE2025 Universal Evaluation

The beauty of X-ARES universality: **ANY existing X-ARES encoder works directly with DCASE2025 tasks!**

## 🎯 True Universality

No need for DCASE-specific encoders. All existing X-ARES encoders work automatically:

### **Existing Encoders That Work Out-of-the-Box**

```bash
# DASHENG encoders
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_universal_task.py
python -m xares.run example/dasheng/dasheng_local_encoder.py src/tasks/dcase2025_autotrash_eval_task.py

# Wav2Vec2 encoders
python -m xares.run example/wav2vec2/wav2vec2_encoder.py src/tasks/dcase2025_universal_task.py

# Whisper encoders
python -m xares.run example/whisper/whisper_encoder.py src/tasks/dcase2025_universal_task.py

# Data2Vec encoders
python -m xares.run example/data2vec/data2vec_encoder.py src/tasks/dcase2025_universal_task.py

# CED encoders
python -m xares.run example/ced/small_ced_pretrained.py src/tasks/dcase2025_universal_task.py
python -m xares.run example/ced/mini_ced_pretrained.py src/tasks/dcase2025_universal_task.py
python -m xares.run example/ced/tiny_ced_pretrained.py src/tasks/dcase2025_universal_task.py
```

## 🏗️ Universal Interface

Every X-ARES encoder already implements the required interface:

```python
class AnyXARESEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Required by X-ARES (automatically works with DCASE)
        self.sampling_rate = 16000      # Target sampling rate
        self.output_dim = 768          # Embedding dimension
        self.hop_size_in_ms = 40.0     # Temporal resolution

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Standard X-ARES format: [B, T] → [B, T', D]
        return self.model(audio)
```

## 📊 DCASE Task Types

### Individual Machine Types
```bash
# Single machine evaluation
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_autotrash_eval_task.py
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_bandsealer_eval_task.py
# ... (8 machine types available)
```

### Universal Task (All Machines)
```bash
# Evaluate all machine types at once
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_universal_task.py
```

## 🎛️ Configuration Options

DCASE tasks support the same X-ARES configuration patterns:

```python
# In dcase2025_universal_task.py
config = DCASETaskConfig(
    name="DCASE2025_Universal",
    encoder=encoder,
    anomaly_method="kmeans",  # DCASE-specific
    n_clusters=10,           # DCASE-specific
    batch_size_train=32,     # Standard X-ARES
    learning_rate=1e-3,      # Standard X-ARES
    output_dim=2,            # Binary classification
    metric="AUC",            # Standard X-ARES
)
```

## 🔄 Comparison Across Encoders

```bash
# Compare different encoders on same DCASE task
for encoder in dasheng wav2vec2 whisper data2vec; do
    echo "Evaluating $encoder on DCASE2025..."
    python -m xares.run example/$encoder/*_encoder.py src/tasks/dcase2025_universal_task.py
done
```

## 📈 Benefits of X-ARES Universal Design

1. **No Encoder Modification**: Existing encoders work unchanged
2. **Consistent Interface**: Same patterns across all audio tasks
3. **Easy Comparison**: Switch encoders without changing evaluation code
4. **Infrastructure Leverage**: Automatic caching, parallel processing
5. **Future-Proof**: New encoders automatically compatible

## 🎉 Example Usage

```bash
# Evaluate DASHENG on DCASE2025 with k-means anomaly detection
python -m xares.run example/dasheng/dasheng_encoder.py src/tasks/dcase2025_universal_task.py

# Results saved in X-ARES standard location:
# env/DCASE2025_Universal/dcase_results/DashengEncoder/teams/baseline/
```

**No special DCASE encoders needed - X-ARES universality works perfectly!** 🚀

---

## Available X-ARES Encoders

### DASHENG
- `example/dasheng/dasheng_encoder.py` - Base DASHENG model
- `example/dasheng/dasheng_local_encoder.py` - Local checkpoint variant

### Wav2Vec2
- `example/wav2vec2/wav2vec2_encoder.py` - Facebook Wav2Vec2

### Whisper
- `example/whisper/whisper_encoder.py` - OpenAI Whisper

### Data2Vec
- `example/data2vec/data2vec_encoder.py` - Meta Data2Vec

### CED (Conformer-based)
- `example/ced/small_ced_pretrained.py` - Small CED model
- `example/ced/mini_ced_pretrained.py` - Mini CED model
- `example/ced/tiny_ced_pretrained.py` - Tiny CED model

All work directly with DCASE2025 tasks - that's the power of universal design!