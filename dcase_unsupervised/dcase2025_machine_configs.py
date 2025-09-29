"""
DCASE2025 Task 2 Individual Machine Type Configurations
Following winning approaches with proper dataset paths from local_dataset_position.txt

These tasks use the two-stage approach:
1. Precompute embeddings from pretrained encoders (no retraining)
2. k-NN anomaly detection using additional dataset as anchors
"""

from .dcase2025_twostage_task import dcase2025_twostage_config, DCASETwoStageConfig


def dcase2025_autotrash_twostage_config(encoder) -> DCASETwoStageConfig:
    """AutoTrash machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="AutoTrash",
        knn_method="kth_distance",  # Following DCASE winners
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_bandsealer_twostage_config(encoder) -> DCASETwoStageConfig:
    """BandSealer machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="BandSealer",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_coffeegrinder_twostage_config(encoder) -> DCASETwoStageConfig:
    """CoffeeGrinder machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="CoffeeGrinder",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_homecamera_twostage_config(encoder) -> DCASETwoStageConfig:
    """HomeCamera machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="HomeCamera",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_polisher_twostage_config(encoder) -> DCASETwoStageConfig:
    """Polisher machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="Polisher",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_screwfeeder_twostage_config(encoder) -> DCASETwoStageConfig:
    """ScrewFeeder machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="ScrewFeeder",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_toypet_twostage_config(encoder) -> DCASETwoStageConfig:
    """ToyPet machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="ToyPet",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


def dcase2025_toyrccar_twostage_config(encoder) -> DCASETwoStageConfig:
    """ToyRCCar machine type with two-stage evaluation"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="ToyRCCar",
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )


# Universal config for all machine types
def dcase2025_universal_twostage_config(encoder) -> DCASETwoStageConfig:
    """Universal config that processes all machine types"""
    return dcase2025_twostage_config(
        encoder=encoder,
        machine_type="Universal",  # Special marker for processing all types
        knn_method="kth_distance",
        k_neighbors=5,
        threshold_method="percentile",
        threshold_percentile=50.0
    )