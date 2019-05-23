import copy

import stanfordnlp

from question_processing import find_q_type, find_q_focus

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


def construct_bigrams(words):
    word_list = [word.text for word in words]
    word_list.insert(0, "ROOT")
    return [
        (word_list[int(word.index)], word.dependency_relation, word_list[word.governor])
        for word in words
    ]


def s(a, b):
    if a == b:
        return 1
    else:
        return 0


def q(a, b, theta=2):
    if a == b:
        return theta
    else:
        return 1


def sim(bigram1, bigram2):
    dep_1, type_1, head_1 = bigram1
    dep_2, type_2, head_2 = bigram2
    
    return (s(dep_1, dep_2) + s(head_1, head_2)) * q(type_1, type_2)


def calculate_tree_distance(sentence, parsed_question):
    sent_bigrams = construct_bigrams(sentence.words)
    quest_bigrams = construct_bigrams(parsed_question.sentences[0].words)

    score = 0
    for sent_bigram in sent_bigrams:
        for quest_bigram in quest_bigrams:
            score += sim(sent_bigram, quest_bigram)

    return score / (len(sent_bigrams) + len(quest_bigrams))


def calculate_focus_score(sentence, question_focus):
    score = 0
    sentence_lemmas = [word.lemma for word in sentence.words]
    for lemma in question_focus:
        if lemma in sentence_lemmas:
            score += 1

    return score / len(question_focus) if question_focus else 0


def calculate_overall_scores(tree_dists, focus_scores):
    return [0.8 * tree_dist + 0.2 * focus_score for tree_dist, focus_score in zip(tree_dists, focus_scores)]


def find_root(sentence):
    for word in sentence.words:
        if word.governor == 0:
            return int(word.index)
    return 0


def find_subject(sentence):
    root_idx = find_root(sentence)
    subjects = []
    if root_idx > 0:
        for word in sentence.words:
            if word.governor == root_idx and word.dependency_relation == "nsubj":
                subjects.append(int(word.index))
                
    return subjects


def find_related_words(sentence, idx):
    idx_ = []

    while idx_ != idx:
        idx_ = copy.deepcopy(idx)
        for word in sentence.words:
            if word.governor in idx and int(word.index) not in idx:
                idx.append(int(word.index))
                
    return idx


def construct_answer_from_idx(sentence, idx):
    return " ".join([word.text for word in sentence.words if int(word.index) in idx])


def check_word_in_question(word, question):
    return word.text.casefold() in question.casefold() 


def construct_sentence(sentence, question):
    return " ".join([word.text for word in sentence.words if not check_word_in_question(word, question) and word.upos != "PUNCT"])


def contains_eachother(word1, word2):
    return word1.text in word2.text or word2.text in word1.text \
           or word1.lemma in word2.lemma or word2.lemma in word1.lemma


def find_left_child(sentence, idx):
    for word in sentence.words:
        if word.governor == idx and word.dependency_relation == "obl":
            return int(word.index)
        
    return -1


def find_relation(sentence, idx, relation):
    related_idx = []
    for word in sentence.words:
        if word.governor == idx and word.dependency_relation == relation:
            related_idx.append(int(word.index))
            
    return related_idx


def entity_answer_extractor(sentence, parsed_question):
    subject_idx = find_subject(sentence)
    related_idx = find_related_words(sentence, subject_idx)
    
    if related_idx:
        return construct_answer_from_idx(sentence, related_idx)
    else:
        return construct_sentence(sentence, parsed_question.text)


def location_answer_extractor(sentence, parsed_question):
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
                return construct_answer_from_idx(sentence, idx)

    return construct_sentence(sentence, parsed_question.text)


def human_answer_extractor(sentence, parsed_question):
    # TODO: Improve this extractor
    return entity_answer_extractor(sentence, parsed_question.text)


def numeric_answer_extractor(sentence, parsed_question):
    # TODO: Improve this extractor
    return construct_sentence(sentence, parsed_question.text)


def reason_answer_extractor(sentence, parsed_question):
    # TODO: Improve this extractor
    return construct_sentence(sentence, parsed_question.text)


def yes_no_answer_extractor(sentence, parsed_question):
    # TODO: Improve this extractor
    return construct_sentence(sentence, parsed_question.text)


def other_answer_extractor(sentence, parsed_question):
    return construct_sentence(sentence, parsed_question.text)


answer_extractors = {
    "Entity": entity_answer_extractor,
    "Location" : location_answer_extractor,
    "Human": human_answer_extractor,
    "Numeric": numeric_answer_extractor,
    "Reason": reason_answer_extractor,
    "YesNo": yes_no_answer_extractor,
    "Other": other_answer_extractor,
}

def get_answer(question_type, sentence, parsed_question):
    answer_extractor = answer_extractors[question_type]
    
    return answer_extractor(sentence, parsed_question)


def extract_answer(passage, question_type, question_focus, parsed_question):
    parsed_pas = nlp(passage)
    tree_dists = []
    for sentence in parsed_pas.sentences:
        tree_dists.append(calculate_tree_distance(sentence, parsed_question))
    
    focus_scores = []
    for sentence in parsed_pas.sentences:
        focus_scores.append(calculate_focus_score(sentence, question_focus))

    scores = calculate_overall_scores(tree_dists, focus_scores)
    idx = scores.index(max(scores))

    answer = get_answer(question_type, parsed_pas.sentences[idx], parsed_question)
    
    return answer


def find_answer(question, passage):
    question_type, question_focus, parsed_question = find_type_focus_parsed(question)
    answer = extract_answer(passage, question_type, question_focus, parsed_question)
    
    return answer


if __name__ == "__main__":
    quest_ = '2015 yılı Birleşmiş Milletler verilerine göre dünyadaki ortalama yaşam süresi kaç yıldır?'
    pass_ = "Ortalama yaşam süresi, ülkelerin sağlıktaki seviyeleri ile doğrudan ilişkili ve gelişmişliğin göstergelerinden biridir. Dünyadaki ortalama yaşam süresi 2015 yılı Birleşmiş Milletler verilerine göre 72 yıldır. Gelişmiş bir ülke olan Japonya'da ortalama yaşam süresinin 84 yıl, az gelişmiş bir ülke olan Nijer'de ise 59 yıl olduğu görülmektedir."

    print(find_answer(quest_, pass_))
