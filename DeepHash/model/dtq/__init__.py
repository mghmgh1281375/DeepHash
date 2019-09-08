from .dtq import DTQ
from .util import Dataset

def train(train_img, database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    img_train = Dataset(train_img, config.output_dim, config.subspace * config.subcenter)
    
    
    #img_train.update_triplets(config.triplet_margin, n_part=config.n_part, select_strategy=config.select_strategy)
    #import os
    #import numpy as np
    #arr = img_train.triplets
    #idx = img_train._perm[np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]], axis=0)]
    #ds = img_train._dataset
    #names = []
    #for i in idx:
    #    a = os.path.join(ds.data_root, ds.lines[i].strip().split()[0])
    #    names.append(a)
    #import pickle as pkl
    #pkl.dump(names, open('/home/sobhe/sam/DeepHash/triples.pkl', 'wb'))
    #np.save('/home/sobhe/sam/DeepHash/triplets.npy', idx)
    
    model.train_cq(img_train, img_query, img_database, config.R)
    return model.save_dir


def validation(database_img, query_img, config):
    model = DTQ(config)
    img_database = Dataset(database_img, config.output_dim, config.subspace * config.subcenter)
    img_query = Dataset(query_img, config.output_dim, config.subspace * config.subcenter)
    return model.validation(img_query, img_database, config.R)
