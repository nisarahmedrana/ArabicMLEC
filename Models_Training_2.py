import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from collections import Counter

# Install necessary packages
get_ipython().system('pip install scikit-multilearn==0.2.0')
get_ipython().system('pip install transformers')
get_ipython().system('pip install pydot graphviz')
get_ipython().system('pip install keras-tuner')

# TensorFlow and Keras
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Add, Concatenate, GlobalMaxPool1D, MaxPooling1D, BatchNormalization, Dropout, SpatialDropout1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

# Transformers
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, AutoModel, TFAutoModel

# Scikit-learn
from sklearn.metrics import f1_score, jaccard_score, classification_report
from sklearn.model_selection import train_test_split, KFold

# Other necessary packages
import numpy as np
import os

# IPython
from IPython.display import Image

AraBERT_tokenizer = AutoTokenizer.from_pretrained("nisarahmedrana/arabert_finetuned_model", from_pt= False)
AraBERT_model = TFAutoModel.from_pretrained("nisarahmedrana/arabert_finetuned_model", from_pt= False)
MARBERT_tokenizer = AutoTokenizer.from_pretrained("nisarahmedrana/marbert_finetuned_model", from_pt= False)
MARBERT_model = TFAutoModel.from_pretrained("nisarahmedrana/marbert_finetuned_model", from_pt= False)
ArabicBERT_tokenizer = AutoTokenizer.from_pretrained("nisarahmedrana/arabic_finetuned_model", from_pt= False)
ArabicBERT_model = TFAutoModel.from_pretrained("nisarahmedrana/arabic_finetuned_model", from_pt= False)



def find_best_thresholds(y_true, y_pred):
    thresholds = np.arange(0, 1, 0.01)
    best_thresholds = []
    for idx in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, idx], y_pred[:, idx] > thresh) for thresh in thresholds]
        best_thresh = thresholds[np.argmax(scores)]
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)


df= pd.read_excel("dataset.xlsx")


df.columns = df.columns.str.strip() # Remove spaces from columns names
print(df.columns)


df.info() #print inforamation about dataframe

# showing Imbalance issue

# Calculate the number of 1s and 0s for each emotion class excluding "neutral"
class_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
counts = []

# Calculating count of 1s and 0s for each class
for col in class_columns:
    counts.append((col, df[col].sum(), len(df[col]) - df[col].sum()))

# Creating a DataFrame for better visualization
counts_df = pd.DataFrame(counts, columns=['Emotion', 'Count_1s', 'Count_0s'])
counts_df


# **preprocessing stage**

# Identify rows where all emotion columns are 0
emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# If all emotion columns are 0, set 'neutral' to 1, otherwise set it to 0
df['neutral'] = df[emotion_columns].sum(axis=1).apply(lambda x: 1 if x == 0 else 0)

# Check if there are any NaN values left in the dataframe
nan_check = df.isna().sum()


df.info() #print inforamation about dataframe


non_string_values = df[df['content'].apply(lambda x: not isinstance(x, str))]
print("Number of non-string values:", len(non_string_values))
non_string_values




# Drop the non-string rows using the updated indices
df = df.drop(non_string_values.index, errors='ignore')




# Recompute the non-string rows
non_string_values = df[df['content'].apply(lambda x: not isinstance(x, str))]




non_string_values




# Exploratory data analysis
# plot the bar plot of labels frequency
y=df.iloc[:,1:].sum()
plt.figure(figsize=(20,12))

labels, counts = y.index, y.values
palette = sns.color_palette("hsv", len(labels))
ax = sns.barplot(x=labels, y=counts, alpha=0.8, palette=palette)


plt.title("Class counts", fontsize=18)
plt.ylabel('# of Occurrences', fontsize=18)
plt.xlabel('Label ', fontsize=25)

rects = ax.patches
labels = y.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=15)

plt.show()
print(y.index)
print(y.values)


# Replace English punctuations in the text with whitespace
def replace_punctuations_with_whitespace(tweet):
    if isinstance(tweet, str):
        return re.sub(r'[!?؟"#$%&\'()*+,\-./:;<=>@\[\]^_`{|}~]', ' ', tweet)
    else:
        return tweet

# Assuming df is your loaded dataframe
df['content'] = df['content'].apply(replace_punctuations_with_whitespace)

# Replace Arabic punctuations in the text with whitespace
def replace_arabic_punctuations(text):
    if isinstance(text, str):
        # Arabic punctuations
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ·•°̣̥ٓ٪ٰ\u200d⁉\u2066\u2067\u2069'''
        # Replace each punctuation with a whitespace
        for punc in arabic_punctuations:
            text = text.replace(punc, " ")
        return text
    else:
        return text

df['content'] = df['content'].apply(replace_arabic_punctuations)


import re


    # Function to replace emojis and emoticons with a space
def replace_emojis_with_space(text):
    if not isinstance(text, str):
        return text

    # Patterns to match emojis and some common emoticons
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric shapes extended
        "\U0001F800-\U0001F8FF"  # Supplemental arrows C
        "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # Chess symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+"
    )
    return emoji_pattern.sub(r' ', text)  # Replacing with space

# Apply the function to replace emojis and emoticons with a space
df['content'] = df['content'].apply(replace_emojis_with_space)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Load the data
train_df = pd.read_excel('dataset.xlsx')

# Define the tokenizer
tokenizer = BertTokenizer.from_pretrained('nisarahmedrana/arabert_finetuned_model')

# Define the dataset class
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 
                              'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['content']
        labels = self.dataframe.iloc[idx][self.label_columns].values.astype(float)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

# Create the dataset
MAX_LENGTH = 128
train_dataset = MultiLabelDataset(train_df, tokenizer, MAX_LENGTH)

# Define the MultiLabelContrastiveFocalLoss class
class MultiLabelContrastiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelContrastiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        focal_loss = focal_weight * bce_loss
        contrastive_loss = self.contrastive_loss(probs, targets)
        total_loss = focal_loss.mean() + contrastive_loss.mean()
        return total_loss

    def contrastive_loss(self, probs, targets):
        batch_size, num_labels = targets.size()
        loss = 0.0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    similarity = torch.dot(probs[i], probs[j])
                    target_similarity = torch.dot(targets[i], targets[j])
                    loss += (1 - target_similarity) * similarity
        loss /= (batch_size * (batch_size - 1))
        return loss

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['content'])

# Define the labels
label_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 
                 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
y_train = train_df[label_columns].values

# Train a simple logistic regression model
model = MultiOutputClassifier(LogisticRegression()).fit(X_train, y_train)

# Get predictions
y_pred = model.predict_proba(X_train)
y_pred = np.array([pred[:, 1] for pred in y_pred]).T

# Calculate the losses
criterion = MultiLabelContrastiveFocalLoss(alpha=0.25, gamma=2.0)
y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
losses = criterion(y_pred_tensor, y_train_tensor)

# Add losses to the dataframe
train_df['loss'] = losses.detach().numpy()

# Analyze loss distribution
loss_threshold = np.percentile(train_df['loss'], 90)  # Select a threshold for rebalancing
hard_samples = train_df[train_df['loss'] > loss_threshold]
balanced_df = pd.concat([train_df, hard_samples])  # Balance by duplicating hard samples

print(f"Original training set size: {len(train_df)}")
print(f"Balanced training set size: {len(balanced_df)}")

def replace_english_and_numbers(text):
    if isinstance(text, str):
        return re.sub(r'[a-zA-Z0-9]', ' ', text)
    else:
        return text
df['content'] = df['content'].apply(replace_english_and_numbers)


def replace_arabic_numbers(text):
    if isinstance(text, str):
        # The range \u0660-\u0669 matches Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)
        return re.sub(r'[\u0660-\u0669]', ' ', text)
    else:
        return text
df['content'] = df['content'].apply(replace_arabic_numbers)

def arabic_normalization(text):
    if not isinstance(text, str):
        return text

    # Replace Arabic letter forms
    text = re.sub("[إأآٱا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub( "چ","ج", text)
    text = re.sub( "ڤ","ف", text)
    text = re.sub( "ڪ","كـ", text)
    text = re.sub( "گ","كـ", text)
    text = re.sub( "ۿ","هـ", text)

    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652]', '', text)

    return text
df['content'] = df['content'].apply(arabic_normalization)


def replace_specific_characters(tweet):
    if not isinstance(tweet, str):
        return tweet

    # Replace the characters "—" and "ღ" with whitespace
    tweet = tweet.replace("—", " ").replace("ღ", " ")

    return tweet

df['content'] = df['content'].apply(replace_specific_characters)

def replace_backslash(text):
    if isinstance(text, str):
        return text.replace("\\", " ")
    else:
        return text

df['content'] = df['content'].apply(replace_backslash)

def remove_repeated_characters(text):
    # This regex finds repeated characters and limits them to a repetition of 2
    return re.sub(r'(.)\1+', r'\1\1', text)



# Ensure that the 'content' column is in string format
df['content'] = df['content'].astype(str)

# Initialize an empty set to store unique words that are longer than 5 characters
long_words = set()

# Function to replace words starting with "وال" with "و ال"
# and reduce repeated characters to a single occurrence
def replace_wal_and_reduce_repeats(text):
    text = re.sub(r'\bوال', 'و ال', text)
    text = re.sub(r'\bهال', 'ه ال', text)
    text = re.sub(r'\bبهال', 'ب ال', text)
    text = re.sub(r'\bلهال', 'ب ال', text)
    text = re.sub(r'\b(.)\1+', r'\1', text)  # reduce repeated characters at the start of a word
    text = re.sub(r'(.)(\1+)\b', r'\1', text)  # reduce repeated characters at the end of a word
    return text

# Apply the function to the 'content' column
df['content'] = df['content'].apply(replace_wal_and_reduce_repeats)

# Iterate through the 'content' column, split into words, and check the length of each word
for content in df['content']:
    for word in content.split():
        if len(word) >= 4:
            long_words.add(word)

def remove_extra_whitespaces(text):
    if isinstance(text, str):
        return re.sub(r'\s+', ' ', text).strip()
    else:
        return text
df['content'] = df['content'].apply(remove_extra_whitespaces)

def remove_single_characters(text):
    # This regex looks for standalone single characters and removes them
    return re.sub(r'\b\w\b', '', text)

    # Apply the new functions
df['content'] = df['content'].apply(remove_single_characters)
df['content'] = df['content'].apply(remove_repeated_characters)



# Embeddings+ Bi-LSTM


# hyperparameters
max_length = 32
batch_size = 32
test_size = 0.05
num_epochs = 50 # set to 30 or 50
lr = 0.001 # learning rate



train, test = train_test_split(df, test_size=test_size, random_state=42)



def bert_encode(data):
    tokens = ArabicBERT_tokenizer.batch_encode_plus(data, max_length=max_length, padding='max_length', truncation=True)

    return tf.constant(tokens['input_ids'])



train_labels = train.drop("content", axis= 1)
test_labels= test.drop("content", axis= 1)



train_encoded = bert_encode(train.content.values.tolist())
test_encoded = bert_encode(test.content.values.tolist())


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_encoded, train_labels))
    .shuffle(10000)
    .batch(batch_size)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((test_encoded, test_labels))
    .shuffle(10000)
    .batch(batch_size)
)



train_encoded


def All_BERT_BiLSTM_model(thresholds=None):
    input_word_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    #last_hidden_states_Ara = AraBERT_model(input_word_ids)[0]
    last_hidden_states_Mar = MARBERT_model(input_word_ids)[0]  # This line is commented out
    last_hidden_states_Arabic = ArabicBERT_model(input_word_ids)[0]
    concat = Concatenate()([last_hidden_states_Mar, last_hidden_states_Arabic])  # Removed last_hidden_states_Mar
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25, dropout=0.3, recurrent_dropout=0.3))(concat)
    x1 = Dense(50, activation='relu')(x)
    if thresholds is not None:
        output = Dense(12, activation=lambda x: tf.keras.activations.sigmoid(x) > thresholds)(x1)
    else:
        output = Dense(12, activation='sigmoid')(x1)
    model = Model(inputs=input_word_ids, outputs=output)
    return model

# Assuming you have defined or will define the AraBERT_model and ArabicBERT_model somewhere in your code
# Example:
# AraBERT_model = ...
# ArabicBERT_model = ...

# Create the model
model = All_BERT_BiLSTM_model()

# Set layer trainability according to your needs
model.layers[1].trainable = False  # set True for fine-tuning the model
model.layers[2].trainable = False  # set True for fine-tuning the model




adam_optimizer = tf.keras.optimizers.legacy.Adam(lr)  # Use the legacy version for M1/M2 Macs #learning rate
model.compile(loss='binary_crossentropy',optimizer=adam_optimizer,metrics=['accuracy'])

model.summary()


#Plot the Model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
Image(retina=True, filename='model_plot.png')


# Assuming 'train_labels' is a DataFrame with your labels
co_occurrence_matrix = train_labels.T.dot(train_labels)

# The matrix is now a DataFrame where each cell [i, j] represents
# how many times label i and label j co-occurred in your dataset

plt.figure(figsize=(12, 10))
sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Label Co-occurrence Matrix")
plt.ylabel("Labels")
plt.xlabel("Labels")
plt.show()


# # Model Initialization



model_checkpoint = ModelCheckpoint(
    filepath = 'saved_models/All_BERT_BiLSTM_model.h5',
    save_weights_only = False,
    monitor = 'val_accuracy',
    mode = 'max',
    verbose = 1,
    save_best_only = True)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose = True) # Early Stopping to prevent overfitting
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.5,
                                                  patience=3,
                                                  min_lr=0.00005,
                                                  verbose=1) # ReduceLROnPlateau Callback to reduce overfitting by decreasing learning




print(train_labels.shape)  # Should print something like (num_samples, 12)
print(test_labels.shape)   # Should print something like (num_samples, 12)


# # Model Training 




All_BERT_BiLSTM_model= model.fit(train_dataset, batch_size=batch_size, epochs=num_epochs, validation_data=test_dataset, callbacks=[model_checkpoint,early,reduce_lr])

# Now compute the best thresholds based on the trained model's predictions on the training data
y_train_pred = model.predict(train_dataset)
best_thresholds = find_best_thresholds(np.array([y for x, y in iter(train_dataset.unbatch())]), y_train_pred)


# Get predictions
y_pred = model.predict(test_dataset)

# Convert predictions to binary format using the thresholds
y_pred_binary = (y_pred > best_thresholds).astype(int)

# Extract true labels
y_true = np.array([y for _, y in test_dataset.unbatch()])

# Calculate metrics
f1_macro = f1_score(y_true, y_pred_binary, average='macro')
f1_micro = f1_score(y_true, y_pred_binary, average='micro')
jaccard = jaccard_score(y_true, y_pred_binary, average='samples')

print("F1 Score (Macro): ", f1_macro)
print("F1 Score (Micro): ", f1_micro)
print("Jaccard Score: ", jaccard)


# Get predictions
y_pred = model.predict(test_dataset)

# Convert predictions to binary format using the thresholds
y_pred_binary = (y_pred > best_thresholds).astype(int)

# Extract true labels
y_true = np.array([y for _, y in test_dataset.unbatch()])

# Define your label names
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
          'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral']

# Calculate and print the classification report
print(classification_report(y_true, y_pred_binary, target_names=labels))


# Assume 'history' is the return value from model.fit()
history = All_BERT_BiLSTM_model

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss Over Time')

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Time')

plt.tight_layout()
plt.show()


# # Model Traning using Cross Validation Kfold=5


# Assuming the function find_best_thresholds is already defined

# Function to create a new dataset
def create_dataset(X, y, indices):
    X_selected = tf.gather(X, indices)
    y_selected = tf.gather(y, indices)
    dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_selected, y_selected))
        .shuffle(10000)
        .batch(batch_size)
    )
    return dataset

# K-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in kf.split(train_encoded):
    train_dataset_fold = create_dataset(train_encoded, train_labels, train_index)
    val_dataset_fold = create_dataset(train_encoded, train_labels, val_index)

    # Reinitialize and compile model for each fold
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    # Fit the model
    print(f'Training for fold {fold_no} ...')
    model.fit(train_dataset_fold, epochs=num_epochs, validation_data=val_dataset_fold, callbacks=[model_checkpoint, early, reduce_lr])

    # Prediction and Evaluation for each fold
    val_encoded, val_labels = tf.gather(train_encoded, val_index), tf.gather(train_labels, val_index)
    y_val_pred = model.predict(val_encoded)
    best_thresholds = find_best_thresholds(val_labels, y_val_pred)
    y_pred_binary = (y_val_pred > best_thresholds).astype(int)

      # Calculate Jaccard Score
    jaccard = jaccard_score(val_labels, y_pred_binary, average=None)
    jaccard_micro = jaccard_score(val_labels, y_pred_binary, average='micro')
    jaccard_macro = jaccard_score(val_labels, y_pred_binary, average='macro')

    print(f'Classification Report for Fold {fold_no}:')
    print(classification_report(val_labels, y_pred_binary, target_names=labels))
    print(f'Jaccard Score for each class for Fold {fold_no}: {jaccard}')
    print(f'Jaccard Score (Micro) for Fold {fold_no}: {jaccard_micro}')
    print(f'Jaccard Score (Macro) for Fold {fold_no}: {jaccard_macro}')

    fold_no += 1

