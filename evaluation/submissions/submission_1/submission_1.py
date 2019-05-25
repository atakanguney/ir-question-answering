import os
import sys
import random
import pickle
from pathlib import Path
from passage_retrieval.paragraph_prediction import analysis
from answer_extraction.answer_extraction import find_answer

SCRIPT_DIR = os.path.dirname(__file__)
PASSAGES_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data/passages.pickle")

_, test_questions_path, task1_pred_path, task2_pred_path = sys.argv

task1_preds, task2_preds = [], []
with open(test_questions_path, 'r', encoding='utf16') as f:
	questions = f.readlines()


with open(PASSAGES_PATH, "rb") as f:
    passages = pickle.load(f)

inverse_passages = {value: key for key, value in passages.items()}
par_list = list(passages.values())
pred_paragraphs = analysis(mode='tf_idf',train_data=par_list, test_data=questions, paragraphs=par_list, n=1, analyzer='word')
task1_preds = [str(inverse_passages[pred]) for pred in pred_paragraphs]
task2_preds = [find_answer(question, passage) for question, passage in zip(questions, pred_paragraphs)]

with open(Path(task1_pred_path) / 'submission_1.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task1_preds))

with open(Path(task2_pred_path) / 'submission_1.txt', 'w', encoding='utf16') as f:
	f.write('\n'.join(task2_preds))
