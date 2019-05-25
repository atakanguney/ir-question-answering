#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:49:22 2019

@author: starlang
"""
import pickle
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from passage_retrieval.paragraph_prediction import (find_similar_paragraphs,
                                                    fit_tf_idf_matrix,
                                                    retrieve_questions,
                                                    transform_matrix)


xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.model")

#%%
def predict_passage(question, passages):
    passage_pars = list(passages.values())
    
    X_vector, X_matrix, X_tf_idf_matrix_df = fit_tf_idf_matrix(passage_pars, 'word')
    
    whole_test = []
    tf_idf_questions = transform_matrix(X_vector, [question])
    similar_paragraphs, similar_paragraph_ids = find_similar_paragraphs(15, tf_idf_questions, X_matrix, passages)
    for s in similar_paragraph_ids:
        tf_idf_pars = transform_matrix(X_vector, [passages[s]])
        whole_matrix = np.concatenate((tf_idf_questions.todense(), tf_idf_pars.todense()), axis = 1)
        whole_test.append(whole_matrix)
        
    X_test = np.array(whole_test)
    nsamples_X_test, nx_X_test, ny_X_test = X_test.shape
    d2_X_test_dataset = X_test.reshape((nsamples_X_test,nx_X_test * ny_X_test))
    
    xgb_pred = xgb_model.predict(d2_X_test_dataset)
    
    y_pred_max = np.argmax(xgb_pred)
    
    return similar_paragraph_ids[y_pred_max]
