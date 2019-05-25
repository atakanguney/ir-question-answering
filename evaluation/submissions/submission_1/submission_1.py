import os
import sys
import random
import pickle
from pathlib import Path

import preprocessing
from passage_retrieval.paragraph_prediction import analysis
from answer_extraction.answer_extraction import find_answer


SCRIPT_DIR = os.path.dirname(__file__)
PASSAGES_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data/passages.pickle")
PASSAGES_STANFORD_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data/passages_stanford.pickle")

_, test_questions_path, task1_pred_path, task2_pred_path = sys.argv

task1_preds, task2_preds = [], []
with open(test_questions_path, 'r', encoding='utf16') as f:
	questions = f.readlines()

with open(PASSAGES_PATH, "rb") as f:
    passages = pickle.load(f)

with open(PASSAGES_STANFORD_PATH, "rb") as f:
    passages_stanford = pickle.load(f)

par_list = list(passages.values())
pred_paragraphs, pred_par_idx = analysis(mode='tf_idf',train_data=par_list, test_data=questions, passages=passages, n=1, analyzer='word')
task2_preds = [find_answer(question, passages_stanford[p_id]) for question, p_id in zip(questions, pred_par_idx)]
task1_preds = [str(id_) for id_ in pred_par_idx]

with open(Path(task1_pred_path) / 'submission_1.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task1_preds))

with open(Path(task2_pred_path) / 'submission_1.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task2_preds))
