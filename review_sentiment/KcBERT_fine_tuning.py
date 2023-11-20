with open('C:/TextminingProject/review_sentiment/review_df_train.txt', encoding='utf-8') as f:
    docs = [doc.strip().split('\t') for doc in f ]
    docs = [(doc[0], int(doc[1])) for doc in docs if len(doc) == 2]
    texts, labels = zip(*docs)

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
model = TFBertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=5, from_pt=True, attention_probs_dropout_prob = 0.5, hidden_dropout_prob = 0.5)

from tensorflow.keras.utils import to_categorical
y_one_hot = to_categorical(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, y_one_hot, test_size=0.3, random_state=2023)

X_train_tokenized = tokenizer(X_train, return_tensors="np", max_length=100, padding='max_length', truncation=True)
X_test_tokenized = tokenizer(X_test, return_tensors="np", max_length=100, padding='max_length', truncation=True)

optimizer = tf.keras.optimizers.legacy.Adam(2e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
checkpoint_filepath = "./review_sentiment/checkpoints/checkpoint_kcbert"
mc = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', mode='min', 
                     save_best_only=True, save_weights_only=True)


history = model.fit(dict(X_train_tokenized), y_train, epochs=500, batch_size=16, 
                    validation_split=0.3, callbacks=[es, mc])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('KcBERT-base')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.savefig('KcBERT-base-loss.png')
plt.clf()

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('KcBERT-base')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.savefig('KcBERT-base-accuracy.png')

import numpy as np
model.load_weights(checkpoint_filepath)
y_preds = model.predict(dict(X_test_tokenized))
prediction_probs = tf.nn.softmax(y_preds.logits,axis=1).numpy()
y_predictions = np.argmax(prediction_probs, axis=1)
y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import classification_report
with open('KcBERT-base', 'w') as text_file:
    print(classification_report(y_predictions, y_test, digits =4), file = text_file)
