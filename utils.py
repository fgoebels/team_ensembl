from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from datetime import datetime

import numpy as np
import pandas as pd



class standardize_text():
    def __init__(self):
        return
    
    def replace_entities(self, col):
        col = col.replace(r'\[WIKI_LINK\:\s[^\]]+\]', 'wikilink', regex=True)
        col = col.replace(r'\[EXTERNA_LINK\:\s[^\]]+\]', 'externalink', regex=True)
        col = col.replace(r'(https?\:\/\/|www)[^\s]+wikipedia[^\s]+', 'wikilink', regex=True)
        col = col.replace(r'(https?\:\/\/|www)[^\s]+[^wikipedia][^\s]+', 'externalink', regex=True)
        
        return col
    
    def clean(self, col):
        col = col.str.lower()
        col = col.replace(r'\n', ' ').replace(r'\t', ' ')
        col = col.replace(r'[^a-z\s]', '', regex=True)
        col = col.replace(r'\s+', ' ', regex=True)
        col = col.replace(r"([a-z]+?)\1+", r"\1\1", regex=True) # removes any repetitions of letters more than twice
        col = col.replace(r"\b(\w+)(\s)(\1\2?)+", r"\1", regex=True) # removes any repetitions of words more than once
        col = col.str.strip()
        
        return col



class create_count_features(BaseEstimator):
    def __init__(self):
        with open("../../External Data/badwords.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = badwords
        
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        ## some handcrafted features!
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split(' ')]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split(' ')])
                                            for c in documents]
        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords_])
                                                for c in documents]
        exclamation = [c.count("!") for c in documents]
        question = [c.count("?") for c in documents]
        newlines = [c.count("\n") for c in documents]
        tabs = [c.count("\t") for c in documents]
        wikilink = [c.count("wikilink") for c in documents]
        extralink = [c.count("externalink") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, 
                         n_chars, 
                         allcaps, 
                         max_word_len,
                         mean_word_len, 
                         exclamation, 
                         question, 
                         tabs,
                         newlines,
                         wikilink, 
                         extralink, 
                         spaces, 
                         bad_ratio, 
                         n_bad, 
                         allcaps_ratio]).T
    
    def get_feature_names(self):
        return ['n_words', 
                 'n_chars', 
                 'allcaps', 
                 'max_word_len',
                 'mean_word_len', 
                 'exclamation', 
                 'question', 
                 'tabs',
                 'newlines',
                 'wikilink', 
                 'extralink', 
                 'spaces', 
                 'bad_ratio', 
                 'n_bad', 
                 'allcaps_ratio']



def train_best_model(pipe, parameters, X_train, y_train, classes):
    predictions = pd.DataFrame()
    models = {}
    score = 0
    for toxicity in classes:
        print('   ' + toxicity)
        est = GridSearchCV(pipe, 
                           parameters, 
                           scoring='roc_auc', 
                           n_jobs=-1,
                           cv=3, 
                           verbose=1,
                           refit=True)
        est.fit(X_train, y_train[toxicity])
        print('best training params: %s' % str(est.best_params_))
        print('best training score: %s' % str(est.best_score_))
        
        score = score + est.best_score_

        models[toxicity] = est.best_estimator_
        
    print('DONE! test mean roc_auc: %s' % str(score/6.))
    return models



def predict_best_model(models, X_train, X_test, classes):
    train_predictions = pd.DataFrame()
    test_predictions = pd.DataFrame()
    for toxicity in classes:
        print('   ' + toxicity)

        train_predictions[toxicity] = models[toxicity].predict_proba(X_train)[:,1]
        test_predictions[toxicity] = models[toxicity].predict_proba(X_train)[:,1]
        
    return train_predictions, test_predictions



def generate_X_stack(model_list):
    all_results = []
    for j, x in enumerate(models_list):
        print(j)
        for tox in classes:
            print('     '+tox)
            all_results.append(x[0][tox].predict_proba(x[1])[:,1])
            
    return np.array(all_results).T



def generate_file_name(prefix='train', initials='AF', description=''):
    return prefix + '_' + initials + datetime.now().strftime('%Y%m%d%H%M') + '_' + description + '.csv'
