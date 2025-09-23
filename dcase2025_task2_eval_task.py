from xares.task import TaskConfig


def dcase2025_task2_eval_config(encoder) -> TaskConfig:
    return TaskConfig(
        batch_size_train=64,
        encoder=encoder,
        eval_weight=2000,
        formal_name="DCASE2025 Task 2 Eval",
        label_processor=lambda x: x["label"],
        learning_rate=1e-3,
        name="DCASE2025_T2_Eval",
        output_dim=2,
        metric="AUC",
        k_fold_splits=list(range(1, 6)),
        zerodo_id=None,
        private=False,
    )


config_audio_tar_name_of_split = {fold: f"wds-audio-fold-{fold}-*.tar" for fold in config.k_fold_splits}
config_encoded_tar_name_of_split = {fold: f"wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits}