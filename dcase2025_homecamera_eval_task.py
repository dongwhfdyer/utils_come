from xares.task import TaskConfig


def dcase2025_homecamera_eval_config(encoder) -> TaskConfig:
    return TaskConfig(
        batch_size_train=64,
        encoder=encoder,
        eval_weight=2000,
        formal_name="DCASE2025 Task 2 Eval - HomeCamera",
        label_processor=lambda x: x["label"],
        learning_rate=1e-3,
        name="DCASE2025_T2_HomeCamera",
        output_dim=2,
        metric="AUC",
        k_fold_splits=list(range(1, 6)),
        zerodo_id=None,
        private=False,
    )


