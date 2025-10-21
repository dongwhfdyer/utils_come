# Dataset Tools

Tools for processing, extending, and managing audio datasets.

## 📁 Scripts Available

| Script | Purpose | Usage |
|--------|---------|-------|
| `extend_esc50.py` | Extend ESC-50 clips from 5 to 10 seconds | `python extend_esc50.py --input_dir ../../datasets/ESC-50/audio --output_dir ../../datasets/ESC-50-extended` |
| `slice_local_file.py` | Slice local video/audio files into clips | `python slice_local_file.py --input ~/Desktop/video.mp4 --output_dir ../../datasets/AudioSet/youtube_sliced_clips` |
| `download_and_slice.py` | Download from YouTube and slice (requires cookies) | `python download_and_slice.py --url "URL" --output_dir ../../datasets/clips` |
| `download_audioset.py` | Download AudioSet clips using audioset-download lib | `python download_audioset.py --num_clips 100` |
| `download_audioset_simple.py` | Simple AudioSet downloader using yt-dlp | `python download_audioset_simple.py --num_clips 100` |

## 🚀 Quick Start

### 1. Extend ESC-50 to 10 seconds

```bash
cd /Users/kuhn/Documents/code/generate_audio_caption/tools/dataset_tools

python extend_esc50.py \
  --input_dir ../../datasets/ESC-50/audio \
  --output_dir ../../datasets/ESC-50-extended \
  --duration 10 \
  --method duplicate
```

**Result:** 2,000 clips × 10 seconds

### 2. Slice a Downloaded Video

```bash
cd /Users/kuhn/Documents/code/generate_audio_caption/tools/dataset_tools

python slice_local_file.py \
  --input ~/Desktop/video.mp4 \
  --output_dir ../../datasets/AudioSet/youtube_sliced_clips \
  --clip_duration 10
```

**Result:** ~360 clips from 1-hour video

### 3. Download from YouTube (if cookies work)

```bash
cd /Users/kuhn/Documents/code/generate_audio_caption/tools/dataset_tools

python download_and_slice.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output_dir ../../datasets/AudioSet/youtube_sliced_clips \
  --clip_duration 10
```

**Note:** YouTube may block automated downloads

## 📖 Script Details

### extend_esc50.py

Extends ESC-50 audio clips from 5 seconds to a target duration.

**Options:**
```
--input_dir       Input directory with ESC-50 audio files
--output_dir      Output directory for extended clips
--duration        Target duration in seconds (default: 10)
--method          Extension method: duplicate, loop, fade (default: duplicate)
```

**Example:**
```bash
python extend_esc50.py \
  --input_dir ../../datasets/ESC-50/audio \
  --output_dir ../../datasets/ESC-50-extended-15sec \
  --duration 15 \
  --method duplicate
```

### slice_local_file.py

Slices a local audio or video file into fixed-duration clips.

**Options:**
```
--input           Input audio or video file (required)
--output_dir      Output directory for clips (default: ./sliced_clips)
--clip_duration   Duration of each clip in seconds (default: 10)
--overlap         Overlap between clips in seconds (default: 0)
--min_duration    Minimum duration for last clip (default: same as clip_duration)
--sample_rate     Target sample rate in Hz (default: 44100)
--format          Output audio format: wav, mp3, flac (default: wav)
```

**Examples:**
```bash
# Basic: 10-second clips
python slice_local_file.py \
  --input ~/Desktop/nature.mp4 \
  --output_dir ../../datasets/nature_clips

# With overlap (more clips)
python slice_local_file.py \
  --input ~/Desktop/music.mp4 \
  --output_dir ../../datasets/music_clips \
  --clip_duration 10 \
  --overlap 2

# Save as MP3 (smaller files)
python slice_local_file.py \
  --input ~/Desktop/ambient.wav \
  --output_dir ../../datasets/ambient_clips \
  --format mp3
```

### download_and_slice.py

Downloads audio from YouTube and slices into clips.

**Options:**
```
--url             Single YouTube URL
--url_file        Text file with multiple URLs (one per line)
--output_dir      Output directory for clips (default: ./sliced_clips)
--clip_duration   Clip length in seconds (default: 10)
--overlap         Overlap between clips in seconds (default: 0)
--temp_dir        Temporary directory for downloads (default: ./temp_downloads)
```

**Note:** Currently blocked by YouTube bot detection. Use `slice_local_file.py` with manually downloaded videos instead.

### download_audioset.py / download_audioset_simple.py

Download individual clips from AudioSet balanced training set.

**Options:**
```
--csv             Path to balanced_train_segments.csv
--output_dir      Output directory for audio files
--num_clips       Number of clips to download (default: 100)
--start           Starting index in CSV (default: 0)
--format          Audio format: wav, mp3, flac (default: wav)
--jobs            Number of parallel downloads (default: 4)
```

**Note:** Many AudioSet YouTube links are broken (30-70% success rate). Manual download + slicing is more reliable.

## 💡 Common Workflows

### Workflow 1: Build a 500-clip dataset from YouTube

1. Manually download 3 videos (30-60 minutes each)
2. Slice each video:
```bash
for video in ~/Desktop/video*.mp4; do
  python slice_local_file.py \
    --input "$video" \
    --output_dir ../../datasets/AudioSet/youtube_sliced_clips
done
```

**Result:** ~500-600 clips

### Workflow 2: Create variations of ESC-50

```bash
# 10-second version
python extend_esc50.py \
  --input_dir ../../datasets/ESC-50/audio \
  --output_dir ../../datasets/ESC-50-extended-10sec \
  --duration 10

# 15-second version
python extend_esc50.py \
  --input_dir ../../datasets/ESC-50/audio \
  --output_dir ../../datasets/ESC-50-extended-15sec \
  --duration 15
```

### Workflow 3: Process multiple downloaded videos

Create a script:
```bash
#!/bin/bash
cd /Users/kuhn/Documents/code/generate_audio_caption/tools/dataset_tools

for video in ~/Desktop/audio_videos/*.mp4; do
  echo "Processing: $video"
  python slice_local_file.py \
    --input "$video" \
    --output_dir ../../datasets/AudioSet/youtube_sliced_clips \
    --clip_duration 10
done

echo "Done! Total clips:"
ls ../../datasets/AudioSet/youtube_sliced_clips/*.wav | wc -l
```

## 🔧 Dependencies

Make sure these are installed:
```bash
pip install librosa soundfile tqdm yt-dlp pandas numpy
```

For video processing:
```bash
brew install ffmpeg  # macOS
```

## 📊 Expected Results

| Script | Input | Output | Time |
|--------|-------|--------|------|
| `extend_esc50.py` | 2,000 × 5-sec clips | 2,000 × 10-sec clips | ~45 seconds |
| `slice_local_file.py` | 1-hour video | ~360 × 10-sec clips | <1 second |
| `slice_local_file.py` | 30-min video | ~180 × 10-sec clips | <1 second |

## 📖 Documentation

For detailed guides, see:
- [Complete Dataset Overview](../../docs/datasets/DATASET_OVERVIEW.md)
- [Quick Start: YouTube Slicing](../../docs/datasets/QUICK_START_YOUTUBE_SLICING.md)
- [Comprehensive Slicing Guide](../../docs/datasets/SLICING_GUIDE.md)
- [Recommended Audio Sources](../../docs/datasets/RECOMMENDED_SOURCES.md)

## 🆘 Troubleshooting

### Issue: ModuleNotFoundError
```bash
pip install librosa soundfile tqdm
```

### Issue: YouTube download fails
Use manual download + `slice_local_file.py` instead of `download_and_slice.py`

### Issue: FFmpeg not found
```bash
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux
```

### Issue: Slow processing
- Use SSD for output directory
- Reduce number of parallel jobs
- Use MP3 format instead of WAV

## 💾 Output Formats

### WAV (default)
- **Pros:** Uncompressed, high quality, no loss
- **Cons:** Large file size (~860 KB per 10-sec clip)
- **Use for:** Training, high-quality requirements

### MP3
- **Pros:** Small file size (~100 KB per 10-sec clip, 8-10x smaller)
- **Cons:** Lossy compression
- **Use for:** Storage-constrained scenarios, testing

### FLAC
- **Pros:** Lossless compression, smaller than WAV
- **Cons:** Larger than MP3
- **Use for:** Balance between quality and size

## 🎯 Tips

1. **For best quality:** Keep original sample rate (usually 44.1 kHz)
2. **For most clips:** Use `--overlap 2` to get 25% more clips
3. **For speed:** Use MP3 format and lower sample rate
4. **For organization:** Use different output directories for different sources

---

**Location:** `/Users/kuhn/Documents/code/generate_audio_caption/tools/dataset_tools/`
