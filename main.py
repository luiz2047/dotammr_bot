import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import telebot;
data = pd.read_csv('db.csv')
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
df = df.drop(np.where(df['Рейтинг в доте на данный момент'] > 12000)[0])
df = df.drop(np.where(df['Максимальный рейтинг в Доте'] > 12000)[0])
df = df.loc[df['Рейтинг в доте на данный момент'] > 100]
df = df.loc[df['Максимальный рейтинг в Доте'] > 100]
df = df.loc[df['Количество часов в игре (хотя бы примерное)'] > 100]
df = df.loc[df['Рейтинг в доте на данный момент'] <= df['Максимальный рейтинг в Доте']]
df = df.loc[df['Количество часов в игре (хотя бы примерное)'] < 40000]
df = df.astype('float32')
df = df.drop('Максимальный рейтинг в Доте',axis=1)

X = df.drop(columns=['Рейтинг в доте на данный момент'])
y = df['Рейтинг в доте на данный момент']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr = RandomForestRegressor(random_state=42)
regr.fit(X,y)

bot = telebot.TeleBot('*APIKEY*')
@bot.message_handler(commands=['start'])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Привет! Если хочешь узнать свой психологический ммр - напиши "Котча"')
@bot.message_handler(content_types=["text"])
def handle_message(message):
    if message.text.strip() == 'Котча' :
        bot.send_message(message.chat.id, 'В данном тесте будет 25 вопросов. \n Ответы на вопросы с 1-ого по 3-ий являются любые целые положительные числа. \n А ответы на остальные вопросы будут целые числа от 1 до 5. \n Где 1 - Совершенно не согласен с утверждением, а 5 - Абсолютно согласен. \n')
        mesg = bot.send_message(message.chat.id,'Напиши - "готов"')
        i = 0
        X_test = []
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
                X_test.append(a)
            i = i + 1
            bot.register_next_step_handler(msg,test,i,X_test)
        elif i == 25:
            a = float(message.text)
            X_test.append(a)
            predictions = regr.predict(np.array(X_test).reshape(1, -1))
            print(predictions)
            pred = predictions[0]
            bot.send_message(message.chat.id, 'Твой психологический ммр -> '+ str(pred))
            print(message.chat.username,'\n', pred)
    except:
        bot.send_message(message.chat.id, 'Ты видимо не понял, как ответить, попробуй еще раз.')
        bot.send_message(message.chat.id, 'Пиши еще раз - "Котча"')
        bot.register_next_step_handler(message,handle_message)

bot.polling(none_stop=True, interval=0)
