"""
DCASE Unsupervised Anomaly Detection Module

This module provides implementations for DCASE Task 2 (First-Shot Unsupervised
Anomalous Sound Detection) following winning technical report approaches.

Key Features:
- Two-stage architecture: embedding precomputation + k-NN anomaly detection
- Frozen pretrained encoder weights (no retraining)
- k=1 nearest neighbor following DCASE winners
- Embedding caching for efficient reuse
- Support for all X-ARES encoders universally

Main Components:
- DCASETwoStageTask: Main two-stage evaluation task
- dcase2025_twostage_config: Configuration factory
- Machine-specific configurations for all 8 DCASE2025 machine types

Usage:
    from dcase_unsupervised import dcase2025_twostage_config, DCASETwoStageTask

    config = dcase2025_twostage_config(encoder, "AutoTrash")
    task = DCASETwoStageTask(config)
    results = task.run()
"""

from .dcase2025_twostage_task import (
    DCASETwoStageConfig,
    DCASETwoStageTask,
    dcase2025_twostage_config,
    create_all_dcase_twostage_configs
)

from .dcase2025_machine_configs import (
    dcase2025_autotrash_twostage_config,
    dcase2025_bandsealer_twostage_config,
    dcase2025_coffeegrinder_twostage_config,
    dcase2025_homecamera_twostage_config,
    dcase2025_polisher_twostage_config,
    dcase2025_screwfeeder_twostage_config,
    dcase2025_toypet_twostage_config,
    dcase2025_toyrccar_twostage_config,
    dcase2025_universal_twostage_config
)

__all__ = [
    # Core classes
    "DCASETwoStageConfig",
    "DCASETwoStageTask",

    # Configuration factories
    "dcase2025_twostage_config",
    "create_all_dcase_twostage_configs",

    # Machine-specific configs
    "dcase2025_autotrash_twostage_config",
    "dcase2025_bandsealer_twostage_config",
    "dcase2025_coffeegrinder_twostage_config",
    "dcase2025_homecamera_twostage_config",
    "dcase2025_polisher_twostage_config",
    "dcase2025_screwfeeder_twostage_config",
    "dcase2025_toypet_twostage_config",
    "dcase2025_toyrccar_twostage_config",
    "dcase2025_universal_twostage_config"
]