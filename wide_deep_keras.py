import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler

COLUMNS = ["UserID", "AnimeID", "UserRating",
           "Genre0", "Genre1", "Genre2", "Genre3", "Genre4", "Genre5", "Genre6", "Genre7", "Genre8", "Genre9", "Genre10",
           "Genre11", "Genre12", "Genre13", "Genre14", "Genre15", "Genre16", "Genre17", "Genre18", "Genre19", "Genre20",
           "Genre21", "Genre22", "Genre23", "Genre24", "Genre25", "Genre26", "Genre27", "Genre28", "Genre29", "Genre30",
           "Genre31", "Genre32", "Genre33", "Genre34", "Genre35", "Genre36", "Genre37", "Genre38", "Genre39", "Genre40",
           "Genre41", "Genre42",
           "MediaType", "Episodes", "OverallRating", "ListMembership"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["UserID", "AnimeID",
                       "Genre0", "Genre1", "Genre2", "Genre3", "Genre4", "Genre5", "Genre6", "Genre7", "Genre8", "Genre9", "Genre10",
                       "Genre11", "Genre12", "Genre13", "Genre14", "Genre15", "Genre16", "Genre17", "Genre18", "Genre19", "Genre20",
                       "Genre21", "Genre22", "Genre23", "Genre24", "Genre25", "Genre26", "Genre27", "Genre28", "Genre29", "Genre30",
                       "Genre31", "Genre32", "Genre33", "Genre34", "Genre35", "Genre36", "Genre37", "Genre38", "Genre39", "Genre40",
                       "Genre41", "Genre42",
                       "MediaType"]
CONTINUOUS_COLUMNS = ["Episodes", "OverallRating", "ListMembership"]

def preprocess(df):
    df[LABEL_COLUMN] = df["UserRating"].apply(lambda x: 1 if x >= 7 else 0)
    df.pop("UserRating")
    y = df[LABEL_COLUMN].values
    df.pop(LABEL_COLUMN)
    
    df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])

    # TODO: select features for wide & deep parts
    
    # TODO: transformations (cross-products)
    # from sklearn.preprocessing import PolynomialFeatures
    # X = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)
    
    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    X = df.values
    return X, y

def main():
    df = pd.read_csv("file:///C:/Users/jaden/Documents/SYDE%20522/Data%20Set/data_user.csv", names = COLUMNS, nrows = 1000)
    train_len = len(df)
    
    X, y = preprocess(df)

    split_perc=0.9
    mask = np.random.rand(len(X)) < split_perc
    X_train = X[mask]
    y_train = y[mask]
    X_test = X[~mask]
    y_test = y[~mask]
    
    wide = Sequential()
    wide.add(Dense(1, input_dim=X_train.shape[1]))
    
    deep = Sequential()
    # TODO: add embedding
    deep.add(Dense(input_dim=X_train.shape[1], output_dim=100, activation='relu'))
    deep.add(Dense(100, activation='relu'))
    deep.add(Dense(50, activation='relu'))
    deep.add(Dense(1, activation='sigmoid'))
    
    model = Sequential()
    model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit([X_train, X_train], y_train, nb_epoch=10, batch_size=32)
    
    loss, accuracy = model.evaluate([X_test, X_test], y_test)
    print('\n', 'test accuracy:', accuracy)
    
if __name__ == '__main__':
    main()
