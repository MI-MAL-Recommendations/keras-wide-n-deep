import pandas as pd
import numpy as np
import time
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from wide_deep_keras import preprocess
from wide_deep_keras import build_wide_deep_model
from wide_deep_keras import COLUMNS

print ("Begin:" + str(time.strftime("%H:%M:%S")))
df = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/data_user.csv", names = COLUMNS, nrows = 100000)
all_data_file = 'all_data.csv'
df.to_csv(all_data_file)
print('All input saved as {}'.format(all_data_file))

X_wide, X_deep, y = preprocess(df)
print ("Preprocess Complete:" + str(time.strftime("%H:%M:%S")))

model = build_wide_deep_model(X_wide.shape[1], X_deep.shape[1])
model.load_weights('keras_model_weights.h5')
    
preds_all = model.predict([X_wide, X_deep])
predict_all_filename = 'all_predictions.csv'
predict_all_file = open(predict_all_filename, 'w')
for item in preds_all:
    predict_all_file.write("%s\n" % item)
print('All predictions saved as {}'.format(predict_all_filename))