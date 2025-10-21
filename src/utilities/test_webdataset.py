"""
Test WebDataset to verify it's correctly formatted for CLAP training.

Usage:
  python src/utilities/test_webdataset.py webdataset/train

This will check the first few samples to ensure:
- Audio files are readable
- JSON metadata is valid
- Caption text is present
"""

import sys
import webdataset as wds
import json
import io


def test_webdataset(dataset_dir, num_samples=5):
    """Test loading WebDataset and display first few samples"""

    print("=" * 60)
    print(f"Testing WebDataset: {dataset_dir}")
    print("=" * 60)

    # Try to load dataset
    try:
        url = f"{dataset_dir}/{{000000..999999}}.tar"
        dataset = wds.WebDataset(url).decode()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    print(f"\nChecking first {num_samples} samples...\n")

    success = True
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        print(f"Sample {i+1}:")
        print(f"  Keys: {list(sample.keys())}")

        # Find audio key
        audio_key = None
        for key in ['wav', 'flac', 'mp3', 'ogg']:
            if key in sample:
                audio_key = key
                break

        if audio_key:
            audio_data, sample_rate = sample[audio_key]
            duration = audio_data.shape[1] / sample_rate
            channels = audio_data.shape[0]
            print(f"  ✓ Audio: {audio_key.upper()}, {sample_rate}Hz, {channels} channel(s), {duration:.2f}s")
        else:
            print(f"  ❌ No audio file found!")
            success = False

        # Check JSON metadata
        if 'json' in sample:
            try:
                if isinstance(sample['json'], bytes):
                    metadata = json.loads(sample['json'].decode('utf-8'))
                else:
                    metadata = json.loads(sample['json'])

                if 'text' in metadata:
                    caption = metadata['text']
                    print(f"  ✓ Caption: '{caption[:80]}{'...' if len(caption) > 80 else ''}'")
                else:
                    print(f"  ❌ 'text' field missing in JSON metadata!")
                    success = False

            except json.JSONDecodeError as e:
                print(f"  ❌ Invalid JSON: {e}")
                success = False
        else:
            print(f"  ❌ No JSON metadata found!")
            success = False

        print()

    print("=" * 60)
    if success:
        print("✓ All samples are valid!")
        print("  Your WebDataset is ready for CLAP training.")
    else:
        print("❌ Some samples have issues.")
        print("  Please check the errors above.")
    print("=" * 60)

    return success


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/utilities/test_webdataset.py <dataset_dir>")
        print("Example: python src/utilities/test_webdataset.py webdataset/train")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    test_webdataset(dataset_dir, num_samples)


if __name__ == "__main__":
    main()
