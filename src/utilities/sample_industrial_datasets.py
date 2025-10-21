"""
Stratified Dataset Sampler for Industrial Audio Domains

This script:
1. Takes multiple dataset paths (different domains)
2. Recursively finds all audio files in each domain
3. Performs stratified random sampling
4. Generates a mapping JSON index file
5. Prepares for CLAP data preparation pipeline

Usage:
  python src/utilities/sample_industrial_datasets.py \
    --config dataset_config.json \
    --output-manifest selected_samples.json \
    --samples-per-domain 100 \
    --total-samples 1000

Or specify domains directly:
  python src/utilities/sample_industrial_datasets.py \
    --domains /server/pump_sounds /server/fan_sounds /server/motor_sounds \
    --output-manifest selected_samples.json \
    --samples-per-domain 100
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


# Supported audio formats
AUDIO_EXTENSIONS = {'.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.ogg', '.OGG'}


def scan_audio_files(domain_path: str, domain_name: str) -> List[Dict]:
    """
    Recursively scan directory for audio files.

    Returns list of dicts with:
    - filepath: absolute path to audio file
    - domain: domain name
    - subdomain: subdirectory structure
    - filename: audio filename
    """
    domain_path = Path(domain_path)
    audio_files = []

    print(f"Scanning domain: {domain_name} ({domain_path})")

    for audio_file in tqdm(domain_path.rglob('*'), desc=f"Scanning {domain_name}"):
        if audio_file.is_file() and audio_file.suffix in AUDIO_EXTENSIONS:
            # Get subdirectory path relative to domain root
            relative_path = audio_file.relative_to(domain_path)
            subdomain = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'

            audio_files.append({
                'filepath': str(audio_file.absolute()),
                'domain': domain_name,
                'subdomain': subdomain,
                'filename': audio_file.name,
                'file_id': audio_file.stem  # filename without extension
            })

    print(f"  Found {len(audio_files)} audio files in {domain_name}")
    return audio_files


def stratified_sample_by_domain(
    all_files: List[Dict],
    samples_per_domain: int = None,
    total_samples: int = None,
    stratify_by_subdomain: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Perform stratified sampling across domains and subdomains.

    Args:
        all_files: List of audio file metadata
        samples_per_domain: Fixed number of samples per domain (optional)
        total_samples: Total samples to select across all domains (optional)
        stratify_by_subdomain: Whether to stratify within subdomains
        seed: Random seed for reproducibility

    Returns:
        List of selected audio file metadata
    """
    random.seed(seed)

    # Group files by domain
    files_by_domain = defaultdict(list)
    for f in all_files:
        files_by_domain[f['domain']].append(f)

    # Determine sampling strategy
    if samples_per_domain is not None:
        # Fixed samples per domain
        target_per_domain = {domain: samples_per_domain for domain in files_by_domain}
    elif total_samples is not None:
        # Proportional to domain size
        total_files = len(all_files)
        target_per_domain = {
            domain: max(1, int(total_samples * len(files) / total_files))
            for domain, files in files_by_domain.items()
        }
    else:
        raise ValueError("Must specify either samples_per_domain or total_samples")

    selected_files = []

    for domain, files in files_by_domain.items():
        target_samples = min(target_per_domain[domain], len(files))

        if stratify_by_subdomain:
            # Stratify by subdomain within this domain
            files_by_subdomain = defaultdict(list)
            for f in files:
                files_by_subdomain[f['subdomain']].append(f)

            # Proportional sampling from each subdomain
            subdomain_samples = {}
            remaining = target_samples

            for subdomain, sub_files in files_by_subdomain.items():
                proportion = len(sub_files) / len(files)
                n_samples = max(1, int(target_samples * proportion))
                n_samples = min(n_samples, len(sub_files), remaining)
                subdomain_samples[subdomain] = n_samples
                remaining -= n_samples

            # Sample from each subdomain
            domain_selected = []
            for subdomain, n_samples in subdomain_samples.items():
                sub_files = files_by_subdomain[subdomain]
                sampled = random.sample(sub_files, n_samples)
                domain_selected.extend(sampled)

            print(f"\n{domain}: Selected {len(domain_selected)} samples")
            print(f"  Subdomain distribution:")
            for subdomain, n in subdomain_samples.items():
                print(f"    {subdomain}: {n} samples")
        else:
            # Simple random sampling
            domain_selected = random.sample(files, target_samples)
            print(f"\n{domain}: Selected {len(domain_selected)} samples")

        selected_files.extend(domain_selected)

    return selected_files


def create_manifest(
    selected_files: List[Dict],
    output_path: str,
    include_stats: bool = True
):
    """
    Create a JSON manifest/index file for selected samples.

    Manifest format:
    {
        "metadata": {
            "total_samples": N,
            "domains": [...],
            "creation_time": "...",
            "statistics": {...}
        },
        "samples": [
            {
                "id": "sample_001",
                "filepath": "/path/to/audio.wav",
                "domain": "pump",
                "subdomain": "normal/speed_1000rpm",
                "filename": "recording_001.wav"
            },
            ...
        ]
    }
    """
    from datetime import datetime

    # Assign unique IDs
    for idx, sample in enumerate(selected_files, start=1):
        sample['id'] = f"sample_{idx:06d}"

    # Calculate statistics
    stats = {}
    if include_stats:
        # Count by domain
        domain_counts = defaultdict(int)
        subdomain_counts = defaultdict(lambda: defaultdict(int))

        for sample in selected_files:
            domain_counts[sample['domain']] += 1
            subdomain_counts[sample['domain']][sample['subdomain']] += 1

        stats = {
            'by_domain': dict(domain_counts),
            'by_subdomain': {
                domain: dict(subdomains)
                for domain, subdomains in subdomain_counts.items()
            }
        }

    # Create manifest
    manifest = {
        'metadata': {
            'total_samples': len(selected_files),
            'domains': list(set(s['domain'] for s in selected_files)),
            'creation_time': datetime.now().isoformat(),
            'statistics': stats
        },
        'samples': selected_files
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Manifest created: {output_path}")
    print(f"  Total samples: {len(selected_files)}")
    print(f"  Domains: {', '.join(manifest['metadata']['domains'])}")
    print(f"{'='*60}")

    return manifest


def load_config(config_path: str) -> Dict:
    """
    Load dataset configuration from JSON file.

    Config format:
    {
        "domains": [
            {
                "name": "pump",
                "path": "/server/datasets/pump_sounds"
            },
            {
                "name": "fan",
                "path": "/server/datasets/fan_sounds"
            }
        ],
        "sampling": {
            "samples_per_domain": 100,
            "stratify_by_subdomain": true,
            "seed": 42
        }
    }
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling of industrial audio datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python sample_industrial_datasets.py --config dataset_config.json --output-manifest selected.json

  # Specifying domains directly
  python sample_industrial_datasets.py \
    --domains /server/pump_sounds /server/fan_sounds /server/motor_sounds \
    --domain-names pump fan motor \
    --samples-per-domain 100 \
    --output-manifest selected.json

  # Total samples with proportional allocation
  python sample_industrial_datasets.py \
    --domains /server/pump_sounds /server/fan_sounds \
    --domain-names pump fan \
    --total-samples 1000 \
    --output-manifest selected.json
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config', type=str, help="JSON config file with domains and sampling settings")
    input_group.add_argument('--domains', nargs='+', help="Domain paths (directories)")

    parser.add_argument('--domain-names', nargs='+', help="Domain names (if using --domains)")

    # Output
    parser.add_argument('--output-manifest', required=True, help="Output JSON manifest file")

    # Sampling strategy
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument('--samples-per-domain', type=int, help="Fixed samples per domain")
    sampling_group.add_argument('--total-samples', type=int, help="Total samples (proportional allocation)")

    # Options
    parser.add_argument('--no-stratify', action='store_true', help="Disable subdomain stratification")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--no-stats', action='store_true', help="Don't include statistics in manifest")

    args = parser.parse_args()

    print("="*60)
    print("Industrial Dataset Stratified Sampler")
    print("="*60)

    # Load configuration
    if args.config:
        print(f"\nLoading config: {args.config}")
        config = load_config(args.config)
        domains_config = config['domains']
        sampling_config = config.get('sampling', {})

        # Extract settings
        domain_paths = [d['path'] for d in domains_config]
        domain_names = [d['name'] for d in domains_config]
        samples_per_domain = sampling_config.get('samples_per_domain', args.samples_per_domain)
        total_samples = sampling_config.get('total_samples', args.total_samples)
        stratify = not sampling_config.get('no_stratify', args.no_stratify)
        seed = sampling_config.get('seed', args.seed)
    else:
        domain_paths = args.domains
        domain_names = args.domain_names or [f"domain_{i+1}" for i in range(len(domain_paths))]
        samples_per_domain = args.samples_per_domain
        total_samples = args.total_samples
        stratify = not args.no_stratify
        seed = args.seed

    # Validate
    if len(domain_names) != len(domain_paths):
        parser.error("Number of domain names must match number of domain paths")

    if samples_per_domain is None and total_samples is None:
        parser.error("Must specify either --samples-per-domain or --total-samples (or in config)")

    # Scan all domains
    print("\nScanning domains...")
    all_files = []
    for domain_name, domain_path in zip(domain_names, domain_paths):
        if not os.path.exists(domain_path):
            print(f"Warning: Domain path not found: {domain_path}")
            continue

        domain_files = scan_audio_files(domain_path, domain_name)
        all_files.extend(domain_files)

    print(f"\nTotal files found: {len(all_files)}")

    if len(all_files) == 0:
        print("Error: No audio files found in any domain!")
        return

    # Perform stratified sampling
    print("\nPerforming stratified sampling...")
    selected_files = stratified_sample_by_domain(
        all_files,
        samples_per_domain=samples_per_domain,
        total_samples=total_samples,
        stratify_by_subdomain=stratify,
        seed=seed
    )

    # Create manifest
    print("\nCreating manifest...")
    manifest = create_manifest(
        selected_files,
        args.output_manifest,
        include_stats=not args.no_stats
    )

    # Print summary
    print("\nSummary by domain:")
    for domain, count in manifest['metadata']['statistics']['by_domain'].items():
        print(f"  {domain}: {count} samples")

    print(f"\nNext steps:")
    print(f"  1. Extract features from selected samples")
    print(f"  2. Generate captions")
    print(f"  3. Run CLAP data preparation pipeline")
    print(f"\nManifest file: {args.output_manifest}")


if __name__ == "__main__":
    main()
