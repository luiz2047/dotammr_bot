import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
data = pd.read_csv("db.csv")
data_row = data.iloc[:, 3:-1]
data_row['Максимальный рейтинг в Доте'] = data_row['Максимальный рейтинг в Доте'].str.extract('(\d+)', expand=False)
data_row['Рейтинг в доте на данный момент'] = data_row['Рейтинг в доте на данный момент'].str.extract('(\d+)', expand=False)
data_row['Количество часов в игре (хотя бы примерное)'] = data_row['Количество часов в игре (хотя бы примерное)'].str.extract('(\d+)', expand=False)
data_row['Какое количество игр в среднем ты играешь в неделю? '] = data_row['Какое количество игр в среднем ты играешь в неделю? '].str.extract('(\d+)', expand=False)
data_row['Возраст'] = data_row['Возраст'].str.extract('(\d+)', expand=False)
df = data_row.fillna(0)
df = df.replace(to_replace = 0, value = 0)
df = df.dropna()
df = df.astype('float32')
df = df.drop(np.where(df['Рейтинг в доте на данный момент'] > 10000)[0])
df = df.drop(np.where(df['Максимальный рейтинг в Доте'] > 10000)[0])
df = df.loc[df['Рейтинг в доте на данный момент'] > 200]
df = df.loc[df['Максимальный рейтинг в Доте'] > 200]
df = df.loc[df['Количество часов в игре (хотя бы примерное)'] > 200]
df = df.loc[df['Рейтинг в доте на данный момент'] <= df['Максимальный рейтинг в Доте']]
df = df.loc[df['Количество часов в игре (хотя бы примерное)'] < 30000]
df = df.astype('float32')
df = df.drop('Максимальный рейтинг в Доте',axis=1)
df['target'] = df['Рейтинг в доте на данный момент']
df = df.drop(columns=['Рейтинг в доте на данный момент'])
train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
[(train_features, label_batch)] = train_ds.take(1)
def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

  # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
all_inputs = []
encoded_features = []

# Numerical features.
for header in ['Возраст', 'Количество часов в игре (хотя бы примерное)']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

categorical_cols = df.columns[2:-1]

for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='float')
    encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='float',
                                               max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer='l2')(all_features)
x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer='l2')(x)
x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer='l2')(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001),
              loss='mean_absolute_error',
              )

history = model.fit(train_ds, epochs=500, verbose=2, validation_data=val_ds, callbacks=[callback, model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

import telebot;
bot = telebot.TeleBot('5596921536:AAGbfPLqGhSjKVE8Py844nuK8iOemfmregc')
@bot.message_handler(commands=['start'])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Привет! Если хочешь узнать свой психологический ммр - напиши "Котча"')
@bot.message_handler(content_types=["text"])
def handle_message(message):
    if message.text.strip() == 'Котча' :
        bot.send_message(message.chat.id, 'В данном тесте будет 25 вопросов. \n Ответы на вопросы с 1-ого по 3-ий являются любые целые положительные числа. \n А ответы на остальные вопросы будут целые числа от 1 до 5. \n Где 1 - Совершенно не согласен с утверждением, а 5 - Абсолютно согласен. \n')
        mesg = bot.send_message(message.chat.id,'Напиши - "готов"')
        i = 0
        X_test = {}
        bot.register_next_step_handler(mesg,test,i,X_test)
    else:
        bot.reply_to(message, 'Не понял тебя, дружок!')
def test(message,i,X_test):
    try:
        if i < 25:
            a = message.text
            msg = bot.send_message(message.chat.id,'Вопрос #'+ str(i+1)+ ':'+'\n' + df.columns[i])
            if i != 0:
                a = float(a)
                X_test[df.columns[i-1]] = a
            i = i + 1
            bot.register_next_step_handler(msg,test,i,X_test)
        elif i == 25:
            a = float(message.text)
            X_test[df.columns[i-1]] = a
            input_dict = {name: tf.convert_to_tensor([value]) for name, value in X_test.items()}
            predictions = model.predict(input_dict)
            print(predictions)
            pred = np.array2string(predictions[0][0])
            bot.send_message(message.chat.id, 'Твой психологический ммр -> '+ pred)
            print(message.chat.username,'\n', pred)
    except:
        bot.send_message(message.chat.id, 'Ты видимо не понял, как ответить, попробуй еще раз.')
        bot.send_message(message.chat.id, 'Пиши еще раз - "Котча"')
        bot.register_next_step_handler(msg,handle_message)
    
bot.polling(none_stop=True, interval=0)