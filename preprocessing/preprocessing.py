import pandas as pd
import numpy as np

def train_validate_test (file_name, test_size=.2):
    """
    Form train, validate, and test set.
    
    :param file_name: File name to open
    :type  file_name: str
    :param test_size: Proportion of test size and validation size
    :type  test_size: float
    :returns: (Train set X, Train set y
               validation set X, validation set y,
               test set X, test set y)
    :rtype:   (numpy.ndarray, numpy.ndarray, 
               numpy.ndarray, numpy.ndarray,
               numpy.ndarray, numpy.ndarray)
    """
    
    import json
    from sklearn.model_selection import StratifiedShuffleSplit
    
    file = open(r'../data/planesnet.json')
    planesnet = json.load(file)
    file.close()
    
    arrays = [np.array (item) for item in planesnet['data']]
    planesnet['data'] = arrays

    df = pd.DataFrame(planesnet)
    df.drop(['locations', 'scene_ids'], axis=1, inplace=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(df, df['labels']):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
        
    prop = strat_test_set['labels'].value_counts()/ len(strat_test_set)
    # Proportions are correct if there are 3 times as many non-planes
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, valid_index in split.split(strat_train_set, strat_train_set['labels']):
        strat_train_set = df.loc[train_index]
        strat_valid_set = df.loc[valid_index]
    
    print ('Size of Train Set is: {}'.format(strat_train_set.shape[0]) )
    print ('Size of Validation Set is: {}'.format(strat_valid_set.shape[0]) )
    print ('Size of Test Set is: {}'.format(strat_test_set.shape[0]) )
        
    X_train = strat_train_set['data'].values
    X_train = np.array(X_train.tolist()).astype(np.float32)
    y_train = strat_train_set['labels'].values.astype(np.float32)
    
    X_valid = strat_valid_set['data'].values
    X_valid = np.array(X_valid.tolist()).astype(np.float32)
    y_valid = strat_valid_set['labels'].values.astype(np.float32)
    
    X_test = strat_test_set['data'].values
    X_test = np.array(X_test.tolist()).astype(np.float32)
    y_test = strat_test_set['labels'].values.astype(np.float32)
    
    return (X_train, y_train,
            X_valid, y_valid,
            X_test, y_test)
    

def inc_pca (X, n_components):
    """
    Perform incremental PCA to reduce dimensions while keeping high variance.
    
    :param X: 2 dim array
    :type  X: numpy.ndarray
    :param n_components: Number of dimensions to be reduced to
    :type  n_components: int
    :return: 2 dim array of reduced dimensions
    :rtype:  numpy.ndarray 
    """

    from sklearn.decomposition import IncrementalPCA
    
    n_batches = 100
    inc_pca = IncrementalPCA (n_components = n_components)
    for X_batch in np.array_split (X, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    return (X_reduced)


def scale (X, scale='constant'):
    """
    Perform normalization.
    
    :param X: 2D array
    :type  X: numpy.ndarray
    :param scale: constant or standard
    :type  scale: str
    :return: 2D numpy array
    :rtype:  numpy.ndarray
    """  
    
    if scale == 'constant':
        X_scaled = X / 255
    elif scale == 'standard':
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # This scales the features to have a mean of zero and unit variance
        pipeline = Pipeline([
            ('std scaler', StandardScaler()),
            ])
        X_scaled = pipeline.fit_transform(X)
    else:
        raise Exception ("Not a scale")
        
    return (X_scaled)