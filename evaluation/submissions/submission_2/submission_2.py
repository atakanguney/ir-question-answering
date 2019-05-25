import os
import sys
import random
import pickle
from pathlib import Path
import xgboost as xgb

import preprocessing
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from passage_retrieval.supervised_xgboost import predict_passage
from answer_extraction.answer_extraction import find_answer


SCRIPT_DIR = os.path.dirname(__file__)
PASSAGES_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data/passages.pickle")
PASSAGES_STANFORD_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data/passages_stanford.pickle")
XGBOOST_MODEL_PATH = os.path.join(SCRIPT_DIR, "xgb_model.model")
print(XGBOOST_MODEL_PATH)

_, test_questions_path, task1_pred_path, task2_pred_path = sys.argv

task1_preds, task2_preds = [], []
with open(test_questions_path, 'r', encoding='utf16') as f:
	questions = f.readlines()

with open(PASSAGES_PATH, "rb") as f:
    passages = pickle.load(f)

with open(PASSAGES_STANFORD_PATH, "rb") as f:
    passages_stanford = pickle.load(f)

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGBOOST_MODEL_PATH)

task1_preds = [predict_passage(question, passages, xgb_model) for question in questions]
task2_preds = [find_answer(question, passages_stanford[p_id]) for question, p_id in zip(questions, task1_preds)]

task1_preds = [str(id_) for id_ in task1_preds]

with open(Path(task1_pred_path) / 'submission_2.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task1_preds))

with open(Path(task2_pred_path) / 'submission_2.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task2_preds))
