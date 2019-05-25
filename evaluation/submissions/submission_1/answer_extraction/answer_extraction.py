import copy
import re

import stanfordnlp

from .question_processing import find_q_type, find_q_focus
from .util import (construct_bigrams,
                   sim,
                   check_word_in_question,
                   find_subject,
                   find_related_words,
                   extract_answer_from_whole_sent,
                   extract_answer_with_idx,
                   find_root,
                   contains_eachother,
                   find_left_child,
                   find_relation,
                   reconstruct_sentence,
                   construct_answer_from_idx,
                   calculate_tree_similarity,
                   calculate_focus_score,
                   calculate_focus_score,
                   construct_sentence)

nlp = stanfordnlp.Pipeline(lang="tr")


def find_type_focus_parsed(question):
    """Find Question Type, Focus, and Parse Question

    Parameters
    ----------
    question: str

    Returns
    -------
    tuple
        Type, Focus, Parsed Question
    """
    q_type = find_q_type(question)
    parsed_q = nlp(question)
    q_focus = find_q_focus(parsed_q)
    q_dep = parsed_q.sentences[0].dependencies
    
    return q_type, q_focus, parsed_q


def entity_answer_extractor(sentence, parsed_question):
    """Extract Answer for Entity Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    subject_idx = find_subject(sentence)
    related_idx = find_related_words(sentence, subject_idx)
    
    if related_idx:
        return extract_answer_with_idx(sentence, related_idx, parsed_question.text)
    else:
        return extract_answer_from_whole_sent(sentence, parsed_question.text)


def description_answer_extractor(sentence, parsed_question):
    # TODO: DocString
    return entity_answer_extractor(sentence, parsed_question)


def location_answer_extractor(sentence, parsed_question):
    """Extract Answer for Location Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    question_sent = parsed_question.sentences[0]
    sent_root_idx = find_root(sentence)
    quest_root_idx = find_root(question_sent)
    root = sentence.words[sent_root_idx - 1]
    
    if contains_eachother(sentence.words[sent_root_idx - 1], question_sent.words[quest_root_idx - 1]):
        if root.upos == "VERB":
            idx = []
            left_child = find_left_child(sentence, sent_root_idx)
            if left_child > 0:
                idx.append(left_child)
                idx = find_related_words(sentence, idx)
            
            amods = find_relation(sentence, sent_root_idx, "amod")
            if amods:
                for i in amods:
                    if i not in idx:
                        idx.append(i)
                
                idx = find_related_words(sentence, idx)
            
            if idx:
                return extract_answer_with_idx(sentence, idx, parsed_question.text)

    return extract_answer_from_whole_sent(sentence, parsed_question.text)


def temporal_answer_extractor(sentence, parsed_question):
    # TODO: DocString
    words = sentence.words
    sent = reconstruct_sentence(sentence)
    if "yıl" in parsed_question.text.casefold() or "yıl" in sent:
        for word in words:
            if word.dependency_relation in ["nummod", "nmod:poss"] and word.upos == "NUM":
                if word.dependency_relation == "nummod":
                    return word.lemma
                else:
                    return word.text
    
    def find_temporal_phrase(temp_keyword):
        if temp_keyword in parsed_question.text.casefold():
            for word in words:
                if temp_keyword in word.text.casefold():
                    pattern = "(\w+\s+" + temp_keyword + ")"
                    pat = re.compile(pattern, re.IGNORECASE)
                    sent = reconstruct_sentence(sentence)
                    all_ = pat.findall(sent)
                    if all_:
                        return all_
                    
        return None
                        
    donem = find_temporal_phrase("dönem")
    if donem:
        return " ".join(donem)
    
    cag = find_temporal_phrase("çağ")
    if cag:
        return " ".join(cag)
    
    yuzyil = find_temporal_phrase("yüzyıl")
    if yuzyil:
        return " ".join(yuzyil)
                
        
    return extract_answer_from_whole_sent(sentence, parsed_question.text)


def human_answer_extractor(sentence, parsed_question):
    """Extract Answer for Human Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    subj = find_subject(sentence)
    if subj:
        return extract_answer_with_idx(sentence, subj, parsed_question.text)
    
    root = find_root(sentence)
    obj = find_relation(sentence, root, "obj")
    if obj:
        return extract_answer_with_idx(sentence, subj, parsed_question.text)

    return extract_answer_from_whole_sent(sentence, parsed_question.text)


def numeric_answer_extractor(sentence, parsed_question):
    """Extract Answer for Numeric Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    candidate = extract_answer_from_whole_sent(sentence, parsed_question.text)
    pat = re.compile(r"(%?\d+)")
    all_ = pat.findall(candidate)
    if all_:
        return " ".join(all_)
    
    return candidate


def yes_no_answer_extractor(sentence, parsed_question):
    """Extract Answer for YesNo Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    for word in sentence.words:
        if "Polarity=Neg" in word.feats:
            return "hayır"
        
    return "evet"


def other_answer_extractor(sentence, parsed_question):
    """Extract Answer for Other Type

    Extracts answer for given `sentence` and `parsed_question`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer
    
    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    return extract_answer_from_whole_sent(sentence, parsed_question.text)


answer_extractors = {
    "Entity": entity_answer_extractor,
    "Location" : location_answer_extractor,
    "Temporal": temporal_answer_extractor,
    "Human": human_answer_extractor,
    "Description": description_answer_extractor,
    "Numeric": numeric_answer_extractor,
    "YesNo": yes_no_answer_extractor,
    "Other": other_answer_extractor,
}

def get_answer(question_type, sentence, parsed_question):
    """Get Answer

    Parameters
    ----------
    question_type: str
        Type of Question
    
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains the answer

    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    answer_extractor = answer_extractors[question_type]
    answer = answer_extractor(sentence, parsed_question)

    if answer:
        return answer
    else:
        return reconstruct_sentence(sentence)


def extract_answer(passage, question_type, question_focus, parsed_question):
    """Extract Answer
    
    Extracts answer from passage for given `question_type`, `question_focus`, `parsed_question` 

    Parameters
    ----------
    passage: stanfordnlp.pipeline.doc.Document
        Passage that consits of sentences

    question_type: str
        Question Type

    question_focus: list
        List of words that are focused on question

    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    tree_sims = []
    for sentence in passage.sentences:
        tree_sims.append(calculate_tree_similarity(sentence, parsed_question))
    
    focus_scores = []
    for sentence in passage.sentences:
        focus_scores.append(calculate_focus_score(sentence, question_focus))

    scores = calculate_overall_scores(tree_sims, focus_scores)
    idx = scores.index(max(scores))

    answer = get_answer(question_type, passage.sentences[idx], parsed_question)
    
    return answer


def find_answer(question, passage):
    """Find Answer
    
    Finds answer of question in `passage`

    Parameters
    ----------
    question: str
        Question

    passage: stanfordnlp.pipeline.doc.Document
        Passage

    Returns
    -------
    str
        Answer
    """
    question_type, question_focus, parsed_question = find_type_focus_parsed(question)
    answer = extract_answer(passage, question_type, question_focus, parsed_question)
    
    return answer
