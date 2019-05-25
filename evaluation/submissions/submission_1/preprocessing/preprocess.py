import pickle
import re

import stanfordnlp
from .util import Answer, Question, QuestionGroup


class StanfordNLPPreprocessor(object):
    """Preprocessor Class

    Preprocess Documents with stanford nlp tool

    Attributes
    ----------
    nlp: stanfordnlp.pipeline.core.Pipeline
        Pipeline object
    """
    def __init__(self, nlp=None, lang="tr"):
        if nlp:
            self.nlp = nlp
        else:
            try:
                self.nlp = stanfordnlp.Pipeline(lang=lang)
            except:
                stanfordnlp.download(lang)
                self.nlp = stanfordnlp.Pipeline(lang=lang)

    def read_data(self, passage_path, question_answer_path, encoding="utf-16"):
        """Reads dataset
        
        Parameters
        ----------
        passage_path: str
            Path to Paragraphs
        question_answer_path: str
            Path to question groups
        encoding: str
            Encoding type
        """
        with open(passage_path, "rb") as f:
            self.passage_plain = f.read().decode(encoding)
        with open(question_answer_path, "rb") as f:
            self.question_groups_plain = f.read().decode(encoding)

    def parse_passage(self, sep="\r\n\r\n"):
        """Parse Passages

        Parameters
        ----------
        sep: str
            Separator for passages
        """
        passages = self.passage_plain.split(sep)
        pattern = re.compile(r"^\d+")
        self.passage_dict = {
            int(pattern.match(passage).group()): self.nlp(passage[pattern.match(passage).end() + 1:].strip())
            for passage in passages
        }

    def parse_question_groups(self, sep="\r\n\r\n"):
        """Parse Question Groups

        Parameters
        ----------
        sep: str
            Separator for question groups
        """
        question_groups_raw = self.question_groups_plain.split(sep)
        q_pat = re.compile(r"^S\d+:")
        a_pat = re.compile(r"^C\d+:")
        rel_pat = re.compile(r"^İlintili Paragraf:")

        self.question_groups = []

        for question_group_raw in question_groups_raw:
            questions = []
            answer = None
            rel_par = None
            for line in question_group_raw.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if q_pat.match(line):
                    idx = q_pat.match(line).group()[:-1]
                    text = self.nlp(line[q_pat.match(line).end() + 1:])
                    questions.append(Question(idx, text))
                elif a_pat.match(line):
                    idx = a_pat.match(line).group()[:-1]
                    text = line[a_pat.match(line).end() + 1:]
                    answer = Answer(idx, text)
                elif rel_pat.match(line):
                    rel_par = int(line[rel_pat.match(line).end() + 1:])
                else:
                    print("Unsupported line: {}: Raw: {}".format(line, question_group_raw))
            self.question_groups.append(
                QuestionGroup(questions, answer, rel_par)
            )

    def save_preprocessed_data(self, passages_path, question_groups_path):
        with open(passages_path, "wb") as f:
            pickle.dump(self.passage_dict, f)

        with open(question_groups_path, "wb") as f:
            pickle.dump(self.question_groups, f)


class Preprocessor(object):
    """Preproces Class"""

    def __init__(self, passages_path):
        self.passages_path = passages_path

    def read_passages(self, encoding="utf-16"):
        with open(self.passages_path, "r", encoding=encoding) as f:
            self.passage_plain = f.read()

    def parse_passages(self, sep="\r\n\r\n"):
        """Parse Passages

        Parameters
        ----------
        sep: str
            Separator for passages
        """
        passages = self.passage_plain.split(sep)
        pattern = re.compile(r"^\d+")
        self.passage_dict = {
            int(pattern.match(passage).group()): passage[pattern.match(passage).end() + 1:].strip()
            for passage in passages
        }

    def save_passages(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.passage_dict, f)
    