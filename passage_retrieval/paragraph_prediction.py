#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:06:28 2019

@author: starlang
"""
import pickle
import sys
sys.path.append("..")

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#%%
def load_passages(file_path):
    """Load passage pickle
    Parameters
    ----------
    file_path: str
        Path of the file that contains passages
    
    Returns
    -------
    dict
        passages
    """
    with open(file_path, "rb") as f:
        passages = pickle.load(f)
    return passages
#%%    
def load_question_groups(file_path):
    """Load question groups pickle
    Parameters
    ----------
    file_path: str
        Path of the file that contains question groups
    
    Returns
    -------
    list
        question groups
    """
    with open(file_path, "rb") as f:
        question_groups = pickle.load(f)
    return question_groups
#%%
def retrieve_paragraphs(passages):
    """Retrieve the paragraphs from passage distionary
    Parameters
    ----------
    passages: dict 
        dictionary of passages with ID and passage mapping
    
    Returns
    -------
    list
        paragraphs
    """
    return [x.text for x in passages.values()]
#%%
def retrieve_questions(question_groups):
    """Retrieve the questions from question_groups
    Parameters
    ----------
    question_groups: list
        list of QuestionGroups
    
    Returns
    -------
    list
        questions
    """
    qa_groups = {}
    qa = []
    for qg in question_groups:
        for i in range(len(qg.questions)):
            qa_groups = { 'q_id' : qg.questions[i].idx, 'q' : qg.questions[i].text.text,'a_id' : qg.answer.idx, 'a' : qg.answer.text, 'rel_par' : qg.related_par_id}
            qa.append(qa_groups)
    return qa
#%%
def fit_count_occurence_matrix(train_data):
    """Fit and transform the Vectorizer as count vectors
    Parameters
    ----------
    train_data: list
        traning data in which counts are calculated
    
    Returns
    -------
    CountVectorizer
        count_vec : count vector
    csr_matrix
        count_occurs : count matrix
    DataFrame
        count_occur_df : dataframe that holds the words and counts
    """
    count_vec = CountVectorizer()
    count_occurs = count_vec.fit_transform(train_data)
    count_occur_df = pd.DataFrame((count, word) for word, count in zip(count_occurs.toarray().tolist()[0], count_vec.get_feature_names()))
    count_occur_df.columns = ['Word', 'Count']
    count_occur_df.sort_values('Count', ascending=False, inplace=True)
    return count_vec, count_occurs, count_occur_df
#%%
def fit_normalized_count_occurrence_matrix(train_data):
    """Fit and transform the Vectorizer as normalized count vectors
    Parameters
    ----------
    train_data: list
        traning data in which counts are calculated
    
    Returns
    -------
    TfidfVectorizer
        norm_count_vec : normalized count vector
    csr_matrix
        count_occurs : normalized count matrix
    DataFrame
        count_occur_df : dataframe that holds the words and normalized counts
    """
    norm_count_vec = TfidfVectorizer(use_idf=False, norm='l2')
    norm_count_occurs = norm_count_vec.fit_transform(train_data)
    norm_count_occur_df = pd.DataFrame((count, word) for word, count in zip(norm_count_occurs.toarray().tolist()[0], norm_count_vec.get_feature_names()))
    norm_count_occur_df.columns = ['Word', 'Count']
    norm_count_occur_df.sort_values('Count', ascending=False, inplace=True)
    return norm_count_vec, norm_count_occurs, norm_count_occur_df
#%%
def fit_tf_idf_matrix(train_data, analyzer='word'):
    """Fit and transform the Vectorizer as tf-idf count vectors
    Parameters
    ----------
    train_data: str
        traning data in which counts are calculated
    analyzer: str
        creates word based or bigram based Vectorizer
    Returns
    -------
    TfidfVectorizer
        norm_count_vec : tf-idf count vector
    csr_matrix
        count_occurs : tf-idf count matrix
    DataFrame
        count_occur_df : dataframe that holds the words and tf-idf values
    """
    tfidf_vec = TfidfVectorizer(analyzer)
    tfidf_count_occurs = tfidf_vec.fit_transform(train_data)
    tfidf_count_occur_df = pd.DataFrame((count, word) for word, count in zip(tfidf_count_occurs.toarray().tolist()[0], tfidf_vec.get_feature_names()))
    tfidf_count_occur_df.columns = ['Word', 'Count']
    tfidf_count_occur_df.sort_values('Count', ascending=False, inplace=True)
    return tfidf_vec, tfidf_count_occurs, tfidf_count_occur_df
#%%
def transform_matrix(vector, test_data):
    """Transform the Vectorizer with the given test data
    Parameters
    ----------
    vector : Vectorizer
        count, norm_count or tf_idf vectors.
    test_data: str
        test data in which counts are calculated
    
    Returns
    -------
    transformed vector
    """
    return vector.transform(test_data)
#%%
def find_similar_paragraphs(n, vector1, vector2, passages):
    """Compares 2 different vectors and find the most similar n paragraphs using cosine similarity
    Parameters
    ----------
    n : int
        number of similar paragraphs
    vector1 : transformed vector
        
    vector2: matrix

    passages : dict
    
    Returns
    -------
    list
        similar_paragraphs
    """
    paragraphs = retrieve_paragraphs(passages)

    similar_paragraphs = []
    similar_paragraph_ids = []
    cosine_similarities = cosine_similarity(vector1, vector2)
    for idx, i in enumerate(cosine_similarities):
        x = cosine_similarities[idx].argsort()[-n:]
        for idx_ in x:
            similar_paragraphs.append(paragraphs[idx_])
            similar_paragraph_ids.append([key  for (key, value) in passages.items() if value.text == paragraphs[idx_]])
    return similar_paragraphs, similar_paragraph_ids
#%%
def analysis(mode, train_data, test_data, paragraphs, n, analyzer):
    """Builds model with given mode and perform analysis according to that mode.
    Parameters
    ----------
    mode : str
        count, norm_count or tf_idf
    train_data: str
        traning data in which train counts are calculated
    test_data: str
        test data in which test counts are calculated
    paragraphs: list
    n : int
        number of similar paragraphs
    analyzer: str
        creates word based or bigram based Vectorizer
    Returns
    -------
    list
        similar_paragraphs
    """
    if mode == 'count':  
        vector, matrix, count_matrix_df = fit_count_occurence_matrix(train_data)
    elif mode == 'norm_count':
        vector, matrix, norm_count_matrix_df = fit_normalized_count_occurrence_matrix(train_data)
    elif mode == 'tf_idf':
        vector, matrix, tf_idf_matrix_df = fit_tf_idf_matrix(train_data, analyzer)
    else:
        raise ValueError('Mode should be either count, norm_count or tf_idf')
        
    transformed_vector = transform_matrix(vector, test_data)
    similar_paragraphs, similar_paragraph_ids = find_similar_paragraphs(n, transformed_vector, matrix, passages)
    return similar_paragraphs, similar_paragraph_ids
#%% 
if __name__ == "__main__":
    passages = load_passages("../preprocessed_data/passages.pickle")
    question_groups = load_question_groups("../preprocessed_data/question_groups.pickle")
    passage_pars = retrieve_paragraphs(passages)
    question_answer = retrieve_questions(question_groups)
    questions = [q['q'] for q in question_answer]
    
    whole_corpus = passage_pars + questions
    
    similar_paragraphs, similar_paragraph_ids = analysis(mode='tf_idf',train_data=passage_pars, test_data=questions, passages=passages, n=1, analyzer='word')
