import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split

data = '/Users/ardapalit/Desktop/archive/chessData.csv'
df = pd.read_csv(data)
df

df = df[pd.to_numeric(df.iloc[:, 1], errors='coerce').notnull()]
df.iloc[:, 1] = df.iloc[:, 1].astype(float)
df.shape

fens = df.iloc[:, 0].values
scores = df.iloc[:, 1].values

def fen_to_matrix(fen):
    rows = fen.split(' ')[0].split('/')
    board = np.zeros((8, 8))
    piece_to_int = {
        'r': -4, 'n': -3, 'b': -2, 'q': -5, 'k': -6, 'p': -1,
        'R': 4, 'N': 3, 'B': 2, 'Q': 5, 'K': 6, 'P': 1
    }
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col] = piece_to_int[char]
                col += 1
    return board

X = np.array([fen_to_matrix(fen) for fen in fens])
y = np.array(scores, dtype=float)

X = X.reshape(-1, 8, 8, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
    Flatten(),
    Dense(64, activation='relu'), #hidden
    Dense(1)  #output
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test)) 