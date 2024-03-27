"""
Requirements:
1. Must use the most minimal arguments possible
2. Use seperate configs for various items
3. `output_dir` should be optional
"""
from dataclasses import dataclass


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir
        overwrite_output_dir

        do_train
        do_eval
        do_predict

        evaluation_strategy
        prediction_loss_only

        auto_find_batch_size
        per_device_train_batch_size
        per_device_eval_batch_size
        group_by_length
        length_column_name

        dataloader_config

        gradient_accumulation_steps
        eval_accumulation_steps
        eval_delay
        eval_steps

        learning_rate
        weight_decay
        adam_beta1
        adam_beta2
        adam_epsilon
        max_grad_norm

        num_train_epochs
        max_steps

        optim
        optim_args

        lr_scheduler_type
        lr_scheduler_kwargs
        warmup_ratio
        warmup_steps

        logging_config
        checkpoint_config
        hub_config

        seed
        data_seed
        full_determinism

        mixed_precision_config
        ddp_config
        fsdp
        fsdp_config
        deepspeed
        accelerator_config

        gradient_checkpointing
        gradient_checkpointing_kwargs

        past_index
        remove_unused_columns
        label_names
        label_smooting_factor
        include_inputs_for_metrics
    """

