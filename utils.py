import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pygame
from keras.models import load_model as keras_load_model
import keras.backend as K

es_callback = EarlyStopping(monitor='val_loss', min_delta=.0001, patience=8, verbose=0, mode='auto')
checkpoint_callback = lambda x, n: ModelCheckpoint(f"asset/models/{path(x, n)}.h5", monitor='val_loss', verbose=1,
                                                save_best_only=True, mode='min')

path = lambda x, n: f"100/{x}{n}"


def save(X, y, i):
    with open(f"asset/data/{path('X', i)}.pickle", 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"asset/data/{path('y', i)}.pickle", 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data():
    with open("asset/data/X.pickle", "rb") as handle:
        X = pickle.load(handle)
    with open("asset/data/y.pickle", "rb") as handle:
        y = pickle.load(handle)
    return X, y


def save_model(model, i):
    model.save(f"asset/models/{path('model', i)}.h5")


def load_model(i=""):
    return keras_load_model(f"asset/models/{path('model', i)}.h5", custom_objects={"r2": r2})


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((750, 750))
    pygame.display.set_caption('SnAIke')
    clock = pygame.time.Clock()

    return screen, clock
