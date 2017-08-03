"""
Functions for text preprocessing, machine learning, hyperparameter searches and diagnosing models
Note: 'tag' means the same thing as 'label'
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import operator

def tokenizer(text):
    """
    Tokenize text
    """
    return text.split()

def tokenizer_porter(text):
    """
    Tokenizes text and stems words using the porter stemmer
    """
    porter=PorterStemmer()
    docs = [porter.stem(word) for word in text.split()]
    return docs

def snowball_stemmer(text):
    """
    Tokenizes text and stems words using the snaowball stemmer
    """
    stemmer=SnowballStemmer("english")
    docs=[stemmer.stem(word) for word in text.split()]
    return docs

def lancaster_stemmer(text):
    """
    Tokenizes text and stems words using the lancaster stemmer
    """
    stemmer=LancasterStemmer()
    docs=[stemmer.stem(word) for word in text.split()]
    return docs

def wordnet_lemmatizer(text):
    """
    Tokenizes text and lemmatizes words using the wordnet lemmatizer
    """
    wnl=WordNetLemmatizer()
    docs=[wnl.lemmatize(word) for word in text.split()]
    return docs

def remove_punctuation(text):
    """
    Get rid of punctuation
    """
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    out = text.translate(table)    
    return out

def get_rid_of_stops(text, stop_list):
    """
    Get rid of stopwords (from stop_list) in text
    """
    return [w for w in text if w not in stop_list]

def tfidf_gridsearch(model, X_train, y_train, stop_list, C_array, all_tokenizers=False):
    """
    GridSearchCV for tfidf preprocessing and model hyperparameter search
    
    Args:
        model: estimator to use- has to have a classes_ attritbute
        X_train: features
        y_train: labels
        C_array: array for C values to test (C is the inverse of regularization strength- smaller values mean stronger regularization). 
        all_tokenizers: Boolean. False means only use tokenizer or tokernizer_porter. 
                        True means test the complete set in tokenizer_list
    Returns:
        Best estimator (pipeline object)
        Best score  
        Best parameters
    """
    
    if all_tokenizers==False:
      tokenizer_list=[tokenizer, tokenizer_porter]
    elif all_tokenizers==True:
      tokenizer_list=[tokenizer, tokenizer_porter, snowball_stemmer, lancaster_stemmer, wordnet_lemmatizer]

    tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        stop_words=stop_list)

    param_grid = [{'vect__ngram_range': [(1, 3)],
                   'vect__tokenizer': tokenizer_list,
                   'clf__C':C_array
                  },              
                  {'vect__ngram_range': [(1, 3)],
                   'vect__tokenizer': tokenizer_list,
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__C':C_array
                  },
                  ]

    model_tfidf = Pipeline([('vect', tfidf),
                            ('clf', model)])

    gs = GridSearchCV(model_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

    gs.fit(X_train, y_train)
    
    print('Best parameter set: %s ' % gs.best_params_)
    print('=================')
    print('CV Accuracy: %.3f' % gs.best_score_)
    
    return gs.best_estimator_, gs.best_score_, gs.best_params_

def plot_confmat(y_test, y_pred):
    """
    Plot confusion matrix
    """
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig,ax = plt.subplots()
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j,y=i,
                   s=confmat[i,j],
                   va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

def get_map_counts(df, tag_id_col, tag_name_col, tag_key_dict):
  """
  Get value counts of df[taf_id_col] and a label map as a series
  tag_key_dict: dictionary, label/tag id as keys, label/tag name as values
  """

  # get value counts with label id as indices
  value_counts=df[tag_id_col].value_counts()
  # get list of label names
  categs=list(set(df[tag_name_col].tolist()))

  # get elems in dicts if in categs
  label_mapper = {k: v for k, v in tag_key_dict.items() if v in categs}
  # label mapper as series, with label id as index
  label_map=pd.Series(label_mapper, name='tag')

  return value_counts, label_map

def classific_report(model, true, pred, label_mapper, 
                    val_counts=None, make_csv=False, filename='classific_report.csv'):
  """
  Args:
        model: model with .classes_ attribute
        true: y_test
        pred: y_pred
        label_mapper: dictionary with tag names as keys, tag IDs as values
        val_counts: pd series. value counts for labels, with labels (numbers) as indices
        make_csv: Boolean, for whether to write to csv or not
        filename: name of filename to write
  Returns:
        classification report as a dataframe
  """
  clf_rep = precision_recall_fscore_support(true, pred)
  out_dict = {
             "precision" :clf_rep[0].round(2)
            ,"recall" : clf_rep[1].round(2)
            ,"f1-score" : clf_rep[2].round(2)
            ,"support" : clf_rep[3]
            }
  out_df = pd.DataFrame(out_dict, index = model.classes_)
  avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
  avg_tot.index = ["avg/total"]
  out_df = out_df.append(avg_tot)
    
  label_map=pd.Series(label_mapper, name='tag')

  out_df=out_df.join(label_map)

  # value counts
  if val_counts is not None:
    val_counts.name='value_counts' # name the column
    out_df=out_df.join(val_counts, how='left')

  if make_csv==True:
    out_df.to_csv(filename, index=True)
  else:
    pass

  return out_df

def show_most_informative_features(vectorizer, clf, n=100):
  """
  Get most informative words, for binary case
  """
  c1=[]
  c2=[]
  feature_names = vectorizer.get_feature_names()
  coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
  top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    
  for (coef_1, fn_1), (coef_2, fn_2) in top:
    l1=(coef_1, fn_1)
    l2=(coef_2, fn_2)
    c1.append(l1)
    c2.append(l2)
  return c1, c2

