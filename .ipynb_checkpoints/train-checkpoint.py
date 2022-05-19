import numpy as np
import os
import tensorflow as tf
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from torch.utils.data import (
    Dataset,
    DataLoader
)
import torch
import torchvision.transforms as transforms
from skimage import io
import logging

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

def train():
    log = logging.getLogger(__name__)

    basedir = os.getcwd()
    datadir = basedir + '/data'

    # reading the csv file with annotated image file names
    train_cultivar= pd.read_csv(datadir + '/train_cultivar_mapping.csv')
    train_cultivar.dropna(inplace=True)

    # turning cultivar labels into strings
    train_cultivar['cultivar']=train_cultivar['cultivar'].astype(str)
    
    # creating list of unique cultivars
    labels=list(np.unique(train_cultivar['cultivar']))
    
    # encoding cultivar_index column
    train_cultivar["cultivar_index"] = train_cultivar["cultivar"].map(lambda item:
                labels.index(item))
    
    # training and validation split 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_cultivar['image'],train_cultivar["cultivar_index"], test_size = 0.2)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # loading model and feature extractor
    model_name_or_path = 'facebook/convnext-tiny-224'
    feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_name_or_path)

    # building feature extractor to grab pixel values
    class FeatureExtractor(object):
        def __call__(self, image, target):
            sample = feature_extractor(image, return_tensors='pt')
            sample["labels"] = target
            return sample

    # building dataset to grab images
    class CultivarDataset(Dataset):
        def __init__(self, df_img, df_label, transform):
            self.labels = df_label
            self.image_path = df_img
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image_path = datadir + '/train_images/' + self.image_path[idx]
            image = io.imread(image_path)

            #y_label = torch.tensor(int(self.labels.iloc[idx]))
            y_label = int(self.labels.iloc[idx])

            data = self.transform(image,y_label)

            return data
    
    # train dataset
    train_ds = CultivarDataset(
        df_img = X_train,
        df_label = y_train,
        transform=FeatureExtractor(),
    )

    # valid dataset
    test_ds = CultivarDataset(
        df_img = X_test,
        df_label = y_test,
        transform=FeatureExtractor(),
    )
    
    from datasets import load_metric

    # argmax finds the greatest probability and assigns label based on max probability
    metric = load_metric("accuracy")
    #def compute_metrics(p):
    #    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # building label and id dicts
    label2id, id2label = dict(), dict()
    for i,label in enumerate(labels):
        label2id[label]=str(i)
        id2label[str(i)]=label
        
    # load model
    model = ConvNextForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )
    
    from transformers import TrainingArguments
    from transformers import EarlyStoppingCallback
    # model arguments
    learning_rates = [5e-4, 2e-3]
    epochs = [3,4,5]
    
    import torch

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'][0] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    def model_init():
        return ConvNextForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id
    )
    
    from transformers import Trainer
    
    for lr in learning_rates:
        for epoch in epochs:
            training_args = TrainingArguments(
              output_dir="./results",
              logging_dir = '/home/runs',
              evaluation_strategy='steps',
              per_device_train_batch_size=64,
              num_train_epochs=epoch,
              save_total_limit = 4, # Only last 4 models are saved
              fp16=True,
              save_steps=100,
              eval_steps=100,
              logging_steps=10,
              learning_rate=lr,
              load_best_model_at_end=True,
            )
    
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                tokenizer=feature_extractor,
                model_init = model_init,
            )
            train_results = trainer.train()
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

            metrics = trainer.evaluate(test_ds)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    #best_run = trainer.hyperparameter_search(n_trials=3, direction="maximize")
    
if __name__ == '__main__':
    train()