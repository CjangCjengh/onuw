from transformers import Trainer, TrainingArguments
from werewolf_belief_model import WerewolfBeliefModel, WerewolfBeliefModelConfig
from data_loader import WerewolfBeliefDataset

train_data_dir = 'train_data'
eval_data_dir = 'eval_data'
cache_dir = 'cache'

config = WerewolfBeliefModelConfig()
model = WerewolfBeliefModel(config)
train_dataset = WerewolfBeliefDataset(train_data_dir, cache_dir)
eval_dataset = WerewolfBeliefDataset(eval_data_dir, cache_dir)

training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=30000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy='steps',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=1000,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=5, 
    report_to=['tensorboard'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=WerewolfBeliefDataset.collate_fn
)

trainer.train()
