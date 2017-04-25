import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

COLUMNS = ["UserID", "AnimeID", "UserRating",
           "Genre0", "Genre1", "Genre2", "Genre3", "Genre4", "Genre5", "Genre6", "Genre7", "Genre8", "Genre9", "Genre10",
           "Genre11", "Genre12", "Genre13", "Genre14", "Genre15", "Genre16", "Genre17", "Genre18", "Genre19", "Genre20",
           "Genre21", "Genre22", "Genre23", "Genre24", "Genre25", "Genre26", "Genre27", "Genre28", "Genre29", "Genre30",
           "Genre31", "Genre32", "Genre33", "Genre34", "Genre35", "Genre36", "Genre37", "Genre38", "Genre39", "Genre40",
           "Genre41", "Genre42",
           "MediaType", "Episodes", "OverallRating", "ListMembership"]
LABEL_COLUMN = "UserRating"
CATEGORICAL_COLUMNS = ["UserID", "AnimeID", "MediaType"]
CONTINUOUS_COLUMNS = ["Episodes", "OverallRating", "ListMembership"]

def preprocess(df):
    y = df[LABEL_COLUMN].values

    genres = []
    for i in range(43):
        genres.append("Genre" + str(i))

    # select features for wide parts
    df_wide = df.filter(genres, axis=1)
    df_wide["UserID"] = df["UserID"]
    df_wide["AnimeID"] = df["AnimeID"]
    df_wide["MediaType"] = df["MediaType"]
    
    # select features for deep parts
    df_deep = df.filter(genres, axis=1)
    df_deep["UserID"] = df["UserID"]
    df_deep["AnimeID"] = df["AnimeID"]
    df_deep["MediaType"] = df["MediaType"]
    df_deep["Episodes"] = df["Episodes"]
    df_deep["OverallRating"] = df["OverallRating"]
    df_deep["ListMembership"] = df["ListMembership"]
    
    df_wide = pd.get_dummies(df_wide, columns=[x for x in CATEGORICAL_COLUMNS])
    df_deep = pd.get_dummies(df_deep, columns=[x for x in CATEGORICAL_COLUMNS])
    y = pd.get_dummies(y).values
    
    # TODO: transformations (cross-products using PolynomialFeatures)

    # Scale all columns equally
    df_wide = pd.DataFrame(MinMaxScaler().fit_transform(df_wide), columns=df_wide.columns)
    df_deep = pd.DataFrame(MinMaxScaler().fit_transform(df_deep), columns=df_deep.columns)

    X_wide = df_wide.values
    X_deep = df_deep.values
    return X_wide, X_deep, y

def main():
    print ("Begin:" + str(time.strftime("%H:%M:%S")))
    df = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/data_user.csv", names = COLUMNS, nrows = 100000)
    print ("Read Complete:" + str(time.strftime("%H:%M:%S")))
    
    X_wide, X_deep, y = preprocess(df)
    print ("Preprocess Complete:" + str(time.strftime("%H:%M:%S")))

    split_perc=0.9
    mask = np.random.rand(len(X_wide)) < split_perc
    X_train_wide = X_wide[mask]
    X_train_deep = X_deep[mask]
    y_train = y[mask]
    X_test_wide = X_wide[~mask]
    X_test_deep = X_deep[~mask]
    y_test = y[~mask]
    
    # Wide network
    wide = Sequential()
    wide.add(Dense(1, input_dim=X_train_wide.shape[1]))
    
    # Deep network
    deep = Sequential()
    # TODO: add embedding
    deep.add(Dense(input_dim=X_train_deep.shape[1], output_dim=100, activation='relu'))
    deep.add(Dense(100, activation='relu'))
    deep.add(Dense(50, activation='relu'))
    deep.add(Dense(1, activation='sigmoid'))
    
    # Combining the two - final layer
    model = Sequential()
    model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print ("Model Complete:" + str(time.strftime("%H:%M:%S")))
    
    model.fit([X_train_wide, X_train_deep], y_train, nb_epoch=10, batch_size=32)
    print ("Fit Complete:" + str(time.strftime("%H:%M:%S")))
    
    loss, accuracy = model.evaluate([X_test_wide, X_test_deep], y_test)
    print('\n', 'test accuracy:', accuracy)
    print ("Evaluate Complete:" + str(time.strftime("%H:%M:%S")))
    
if __name__ == '__main__':
    main()
