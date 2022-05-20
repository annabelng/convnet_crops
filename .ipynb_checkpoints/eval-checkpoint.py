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
from transformers import ViTFeatureExtractor, ViTForImageClassification

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

def predict():
    
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

    # building label and id dicts
    label2id, id2label = dict(), dict()
    for i,label in enumerate(labels):
        label2id[label]=str(i)
        id2label[str(i)]=label
    
    # reading the csv file with annotated image file names
    test_cultivar= pd.read_csv(datadir + '/sample_submission.csv')
    test_cultivar.dropna(inplace=True)

    # creating list of unique cultivars
    labels=list(np.unique(test_cultivar['cultivar']))

    # turning cultivar labels into strings
    test_cultivar['cultivar']=test_cultivar['cultivar'].astype(str)
    test_cultivar["cultivar_index"] = test_cultivar["cultivar"].map(lambda item: labels.index(item))

    model_name_or_path = basedir + '/results/vit'
    #feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_name_or_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    # building feature extractor to grab labels
    class FeatureExtractor(object):
        def __call__(self, image, target):
            sample = feature_extractor(image, return_tensors='pt')
            sample["labels"] = target
            return sample

    class CultivarDataset(Dataset):
        def __init__(self, df_img, df_label, transform):
            self.labels = df_label
            self.image_path = df_img
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image_path = datadir + '/test/' + self.image_path[idx]
            image = io.imread(image_path)

            y_label = torch.tensor(int(self.labels.iloc[idx]))
           # y_label = int(self.labels.iloc[idx])

            data = self.transform(image,y_label)
            # data['pixel_values'] = torch.squeeze(data['pixel_values'])
            return data
        
    test_ds = CultivarDataset(
    df_img = test_cultivar['filename'],
    df_label = test_cultivar['cultivar_index'],
    transform = FeatureExtractor())
    
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'][0] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    from transformers import Trainer
    from transformers import TrainingArguments
    #model = ConvNextForImageClassification.from_pretrained(model_name_or_path)
    model = ViTForImageClassification.from_pretrained(
            model_name_or_path
        )
    model.eval()
    
    from datasets import load_metric
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model=model,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        
    )
    
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    pred = trainer.predict(test_ds).predictions
    
    with open(basedir + '/vitpredictions','w') as f:
    for row in pred:
        f.write(str(row))
        f.write('\n')
        
    from scipy.special import softmax
    # softmax each row so each row sums to 1
    prob = softmax(pred, axis = 1)
    culti_index = np.argmax(prob,axis =1)
    
    labels = []
    for val in culti_index:
        key_val = str(val)
        labels.append(id2label[key_val])
        
    submission = pd.DataFrame()
    submission['filename'] = test_cultivar['filename']
    submission['cultivar'] = labels
    submission.drop([0])
    submission.to_csv(basedir+'/vitsubmission.csv', index=False)
    
if __name__ == '__main__':
    predict()   
