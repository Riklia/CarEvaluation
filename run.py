import torch
from torch import nn
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import utils
from model import LinearModel, metrics
import engine
import data_loader

os.makedirs(utils.params.model_dir, exist_ok=True)

headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
cars_df = pd.read_csv(r'data\Car Evaluation\car.data', header=None,
                     names=headers)

df = cars_df.copy()
label_encoder = LabelEncoder()

for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop(columns=['class']).to_numpy()
y = df['class'].tolist()

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# scaler = MinMaxScaler()
# scaler.fit(X_resampled)
#
# X_norm = scaler.transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.33, random_state=42, stratify=y_resampled)


torch.manual_seed(42)
if utils.params.cuda:
    torch.cuda.manual_seed(42)

device = 'cuda' if utils.params.cuda else None

utils.set_logger(os.path.join(utils.params.model_dir, 'train.log'))

logging.info("Loading the datasets...")

train_dataset = data_loader.CarEvaluationDataset(X_train, y_train)
test_dataset = data_loader.CarEvaluationDataset(X_test, y_test)

train_dataloader = data_loader.DataLoader(train_dataset,
                              batch_size=utils.params.batch_size,
                              shuffle=True
                             )

test_dataloader = data_loader.DataLoader(test_dataset,
                              batch_size=utils.params.batch_size,
                              shuffle=False
                            )

logging.info("- done.")

model = LinearModel(in_dim=X.shape[1], out_dim=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=utils.params.learning_rate)

loss_fn = nn.MSELoss().to(device)

logging.info("Starting training for {} epoch(s)".format(
    utils.params.num_epochs))
train_stats, val_stats = engine.train_and_evaluate(model, train_dataloader,
                                             test_dataloader, optimizer, loss_fn, metrics,
                                                   utils.params)

train_stats = {k: [dic[k] for dic in train_stats] for k in train_stats[0]}
val_stats = {k: [dic[k] for dic in val_stats] for k in val_stats[0]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))

ax1.plot(train_stats['loss'])
ax1.plot(val_stats['loss'])

ax1.set_title('Loss')
ax1.legend(['Train', 'Validation'])


ax2.plot(train_stats["f1_score_macro"])
ax2.plot(val_stats["f1_score_macro"])

ax2.set_title("f1 score (macro)")
ax2.legend(["Train", "Validation"], loc="lower left")

plt.savefig(os.path.join(utils.params.model_dir, "results.png"))

print(f'Maximum validation score: {max(val_stats["f1_score_macro"]):0.3f} '
      f'at epoch {np.argmax(val_stats["f1_score_macro"])}')
