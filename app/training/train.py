from torch.utils.data import DataLoader
from sentence_transformers import losses
from sbert.Sentence_Transformer import custom_SentenceTransformer
from sbert.evaluator import custom_EmbeddingSimilarityEvaluator
import time
from get_data import Iterable_dataset, Map_dataset
import os


def give_dataloader_loss_evaluator(train_examples_, val_examples_, model_, batch_size_=64, val_batch_size_=64):
    training_dataloader = DataLoader(train_examples_, batch_size=batch_size_, num_workers=4, shuffle=True)
    validation_dataloader = DataLoader(val_examples_, batch_size=val_batch_size_, num_workers=4)
    loss_ = losses.MultipleNegativesRankingLoss(model=model_)
    evaluator_ = custom_EmbeddingSimilarityEvaluator()
    return training_dataloader, validation_dataloader, loss_, evaluator_


def train(path_, model_, train_dataloader_, val_dataloader_, train_loss_, my_evaluator_, num_epochs_=5,
          steps_=5000, batch_size_=64, dataset_size_=10000000, base_model_='all_miniLM_L6_v2'):
    warmup_steps = int(len(train_dataloader_) * num_epochs_ * 0.1)  # 10% of train data
    print('Training started.....')
    model_.fit(train_objectives=[(train_dataloader_, train_loss_)],
               validation_objectives=[val_dataloader_],
               epochs=num_epochs_,
               evaluator=my_evaluator_,
               evaluation_steps=steps_,
               warmup_steps=warmup_steps,
               save_best_model=True,
               output_path=path_,
               batch_size=batch_size_,
               dataset_size=dataset_size_,
               base_model=base_model_)
    return model_


DATA_DIR = '/home/ubuntu/data'

# print('-------------------------------------------------------------')
# List files in the directory (excluding subdirectories)
# file_list = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
# print(os.getcwd())
# Print the list of files
# print(file_list)
# print('---------------------------------------------------------------')

# Model Load
s = time.time()
print('Model loading...')
model = custom_SentenceTransformer('all-MiniLM-L6-v2')
print(f'Model load finished={time.time() - s}')

# Data Load
s = time.time()
print('Dataset Object loading...')
# train_data = Iterable_dataset(path=DATA_DIR + '/train_data_10M.csv', length=10000)
# val_data = Iterable_dataset(path=DATA_DIR + '/val_data.csv', length=10000)
train_data = Map_dataset(path=DATA_DIR + '/train_data_10M.csv')
val_data = Map_dataset(path=DATA_DIR + '/val_data.csv')
print(f'Dataset Object finished={time.time() - s}')

# Hyperparameters
batch_size = 128
val_batch_size = 128
epochs = 3

# Data Loaders, Loss and Evaluators
train_dataloader, val_dataloader, loss, evaluator = give_dataloader_loss_evaluator(train_data, val_data, model,
                                                                                   batch_size_=batch_size,
                                                                                   val_batch_size_=val_batch_size)
evaluation_steps = int(0.1 * len(train_dataloader))

# Training
model = train('/home/ubuntu/checkpoints/models_checkpoints/entity_search/saved_models/Finetuned_miniLM_10M', model,
              train_dataloader, val_dataloader, loss, evaluator,
              num_epochs_=epochs,
              steps_=evaluation_steps,
              batch_size_=batch_size,
              dataset_size_=10000000,
              base_model_='all_miniLM_L6_v2')
