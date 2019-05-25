#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:09:13 2019

@author: starlang
"""
#%%
from importlib import reload
import passage_retrieval

reload(passage_retrieval)

#%%
import pickle

import preprocessing
from passage_retrieval.paragraph_prediction import analysis
from answer_extraction.util import calculate_tree_similarity

#%%
PASSAGES_PATH = "preprocessed_data/passages.pickle"
PASSAGES_STANFORD_PATH = "preprocessed_data/passages_stanford.pickle"

with open(PASSAGES_PATH, "rb") as f:
    passages = pickle.load(f)
    
with open(PASSAGES_STANFORD_PATH, "rb") as f:
    passages_stan = pickle.load(f)
    
#%%
QUESTION_GROUPS_PATH = "preprocessed_data/question_groups_stanford.pickle"

with open(QUESTION_GROUPS_PATH, "rb") as f:
    qg_stan = pickle.load(f)
    
#%%
q_dict = {}
for qg in qg_stan:
    qs = qg.questions
    if not qg.related_par_id:
        continue
    for q in qs:
        q_dict[q.idx] = q.text, qg.answer.text, qg.related_par_id

#%%
q_idx = list(q_dict.keys())
sample = q_idx[:]

questions = [q_dict[id_][0].text for id_ in sample]

#%%
similar_pars, similar_pars_idx = analysis("tf_idf", list(passages.values()), questions, passages, 5, "word")

pred_pars = []
true_pars = []
for i, q_id in enumerate(sample):
    q_, a, rp_id = q_dict[q_id]
    
    similar_pars_idx_ = similar_pars_idx[i*5: (i+1)*5]
    par_scores = []
    for p_id in similar_pars_idx_:
        scores = []
        for sent in passages_stan[p_id].sentences:
            scores.append(calculate_tree_similarity(sent, q_))
    
        par_scores.append(max(scores))

    argmax = par_scores.index(max(par_scores))
    pred_pars.append(similar_pars_idx_[argmax])
    true_pars.append(rp_id)
#%%
similar_pars, similar_pars_idx = analysis("tf_idf", list(passages.values()), questions, passages, 1, "word")

#%%
np.equal(similar_pars_idx, true_pars).sum() / len(similar_pars_idx)     
#%%
import numpy as np
#%%
np.equal(pred_pars, true_pars).sum() / len(pred_pars)     