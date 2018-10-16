import os
import pickle

import numpy as np
import pandas as pd

from itertools import combinations, product

import keras
import keras.backend as K

from keras.layers import (Add, Input, Dense, BatchNormalization,
                          Activation, Dropout, Embedding, Lambda)

from keras.callbacks import ModelCheckpoint, CSVLogger

from keras.layers import Subtract
from keras.layers import Activation

from keras.models import Model

from nfp.models import create_GRU_embedding_model, masked_mean_absolute_error
from nfp.layers import GraphOutput

data = pd.concat([
    pd.read_csv('20180703_new_nitrogenated_compounds.csv'),
    pd.read_csv('20180720_acetyl_ysis.csv'),
    pd.read_csv('ysi.csv'),
], sort=False).drop_duplicates(keep='first')

test = data.sample(frac=0.1, random_state=0)
valid = data[~data.index.isin(test.index)].sample(n=len(test), random_state=0)
train = data[~data.index.isin(test.index) & ~data.index.isin(valid.index)].sample(frac=1, random_state=0)

from nfp.preprocessing import SmilesPreprocessor, get_max_atom_bond_size

def mean_absolute_percentage_error2(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(
        K.abs(y_true), 10, None))
    return 100. * K.mean(diff, axis=-1)


def pairwise_comparison(inputs, y):
    
    pairwise_indexer = np.array(list(combinations(range(len(y)), 2)))
    
    pairwise_inputs = {}
    pairwise_inputs['atom1'] = inputs['atom'][pairwise_indexer[:, 0]]
    pairwise_inputs['atom2'] = inputs['atom'][pairwise_indexer[:, 1]]

    pairwise_inputs['connectivity1'] = inputs['connectivity'][pairwise_indexer[:, 0]]
    pairwise_inputs['connectivity2'] = inputs['connectivity'][pairwise_indexer[:, 1]]

    pairwise_y = (y.values[pairwise_indexer[:, 0]] - 
                  y.values[pairwise_indexer[:, 1]])
    
    return pairwise_inputs, pairwise_y


def pairwise_comparison_valid(inputs_train, y_train, inputs_valid, y_valid):
    
    valid_indexer = np.array(list(product(range(len(train)), range(len(valid)))))

    pairwise_valid_inputs = {}
    pairwise_valid_inputs['atom1'] = inputs_train['atom'][valid_indexer[:, 0]]
    pairwise_valid_inputs['atom2'] = inputs_valid['atom'][valid_indexer[:, 1]]

    pairwise_valid_inputs['connectivity1'] = inputs_train['connectivity'][valid_indexer[:, 0]]
    pairwise_valid_inputs['connectivity2'] = inputs_valid['connectivity'][valid_indexer[:, 1]]

    pairwise_valid_y = (y_train.values[valid_indexer[:, 0]] - 
                        y_valid.values[valid_indexer[:, 1]])
    
    return pairwise_valid_inputs, pairwise_valid_y

y_test = test.YSI
y_valid = valid.YSI
y_train = train.YSI

if __name__ == "__main__":

    preprocessor = SmilesPreprocessor(
        max_atoms=55, max_bonds=55, max_degree=4, explicit_hs=True,
        feature_set='v2')

    inputs_train = preprocessor.fit(train.SMILES)
    inputs_valid = preprocessor.predict(valid.SMILES)
    inputs_test = preprocessor.predict(test.SMILES)
    
    # Define Keras model
    # Define shared model

    def atom_model():

        # Raw (integer) graph inputs
        atom_types = Input(
            shape=(preprocessor.max_atoms,), name='atom', dtype='int32')
        connectivity = Input(
            shape=(preprocessor.max_atoms * preprocessor.max_degree, 3),
            name='connectivity', dtype='int32')

        atom_embedding_model = create_GRU_embedding_model(
            atom_features=32,
            message_steps=4,
            message_dropout=0.0,
            atom_classes=preprocessor.atom_classes,
            bond_classes=preprocessor.bond_classes,
            max_atoms=preprocessor.max_atoms,
            max_degree=preprocessor.max_degree)

        atom_embedded = atom_embedding_model([atom_types, connectivity])
        fingerprint = GraphOutput(
            128, activation='sigmoid', dropout=0.0
        )([atom_embedded, atom_types])
        # fingerprint = BatchNormalization(center=False)(fingerprint)

        return Model([atom_types, connectivity], [fingerprint], name='molecule_model')


    atom_types1 = Input(
        shape=(preprocessor.max_atoms,), name='atom1', dtype='int32')
    connectivity1 = Input(
        shape=(preprocessor.max_atoms * preprocessor.max_degree, 3),
        name='connectivity1', dtype='int32')

    atom_types2 = Input(
        shape=(preprocessor.max_atoms,), name='atom2', dtype='int32')
    connectivity2 = Input(
        shape=(preprocessor.max_atoms * preprocessor.max_degree, 3),
        name='connectivity2', dtype='int32')

    model = atom_model()

    embed_1 = model([atom_types1, connectivity1])
    embed_2 = model([atom_types2, connectivity2])

    X = Subtract()([embed_1, embed_2])
    X = Activation('tanh')(X)
    X = BatchNormalization(center=False)(X)
    X = Dropout(0.5)(X)

    # X = Dense(128, use_bias=False, activation='tanh')(X)
    X = Dense(1, use_bias=False)(X)

    lr = 1E-3
    epochs = 100
    decay = lr/epochs

    model = Model([atom_types1, connectivity1, atom_types2, connectivity2], [X])
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, decay=decay),
                  loss=mean_absolute_percentage_error2)
    model.summary()

    model_name = 'small_drop'

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    with open(model_name + '/preprocessor.p', 'wb') as f:
        pickle.dump(preprocessor, f)

    filepath= model_name + "/best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True, period=10, verbose=1)
    csv_logger = CSVLogger(model_name + '/log.csv')

    hist = model.fit(
        *pairwise_comparison(inputs_train, y_train),
        validation_data=pairwise_comparison_valid(inputs_train, y_train, inputs_valid, y_valid),
        epochs=epochs,
        verbose=1, shuffle=True, batch_size=500,
        callbacks=[checkpoint, csv_logger])
