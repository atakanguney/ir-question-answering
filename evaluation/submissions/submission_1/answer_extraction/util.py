import copy


def find_root(sentence):
    """Find Root Index
    
    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence

    Returns
    -------
    int
        Greater than 0 if sentence has root
        0 otherwise
    """
    for word in sentence.words:
        if word.governor == 0:
            return int(word.index)
    return 0


def find_relation(sentence, idx, relation):
    """Find Related Word

    Find related word for given `idx` for given `relation`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains answer

    idx: int
        Index for base word

    relation: str
        Dependency relation

    Returns
    -------
    list
        list of related words    
    """
    related_idx = []
    for word in sentence.words:
        if word.governor == idx and word.dependency_relation == relation:
            related_idx.append(int(word.index))
            
    return related_idx


def find_related_words(sentence, idx):
    """Find Related Words
    
    Finds related words which head in the tree is in idx

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence

    idx: list
        List of indices

    Returns
    -------
    list
        List of enhanced indices
    """
    idx_ = []

    while idx_ != idx:
        idx_ = copy.deepcopy(idx)
        for word in sentence.words:
            if word.governor in idx and int(word.index) not in idx:
                idx.append(int(word.index))
                
    return idx


def contains_eachother(word1, word2):
    """Check Containing of 2 words
    
    Checks whether word1 is contained in word2 or word2 is contained in word1

    Parameters
    ----------
    word1: stanfordnlp.pipeline.doc.Word

    word2: stanfordnlp.pipeline.doc.Word

    Returns
    -------
    True if word1 is contained by word2 or otherwise
    False otw.
    """
    return word1.text in word2.text or word2.text in word1.text \
           or word1.lemma in word2.lemma or word2.lemma in word1.lemma


def find_left_child(sentence, idx):
    """Find Left Child
    
    Finds left child of element which has index `idx`

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that has the answer

    idx: int

    Returns
    -------
    int
        Index of left child
    """
    for word in sentence.words:
        if word.governor == idx and word.dependency_relation == "obl":
            return int(word.index)
        
    return -1


def construct_answer_from_idx(sentence, idx):
    """Construct Answer from Index List
    
    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence

    idx: list
        List of indices

    Returns
    -------
    str
        Answer
    """
    return " ".join([word.text for word in sentence.words if int(word.index) in idx])


def find_child(sentence, idx, relation):
    # TODO: Doc-String
    for word in sentence.words:
        if word.governor == idx and word.dependency_relation == relation:
            return int(word.index)
    
    return -1


def find_subject(sentence):
    """Find Subjects of Sentence
    
    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence

    Return 
    ------
    list
        Indices 
    """
    root_idx = find_root(sentence)
    subjects = []
    if root_idx > 0:
        for word in sentence.words:
            if word.governor == root_idx and word.dependency_relation == "nsubj":
                subjects.append(int(word.index))
                
    return subjects


def construct_bigrams(words):
    """Construct Bigrams

    Parameters
    ----------
    words: list
        List of words

    Returns
    -------
    list
        List of triples; (dependent, relation, head)
    """
    word_list = [word.text for word in words]
    word_list.insert(0, "ROOT")
    return [
        (word_list[int(word.index)], word.dependency_relation, word_list[word.governor])
        for word in words
    ]


def s(a, b):
    """Binary 's' function

    Formulation has been taken from [https://www.cmpe.boun.edu.tr/~ozgur/papers/617_Paper.pdf]

    Parameters
    ----------
    a: str
    b: str

    Returns
    -------
    1 if a equals b
    0 otw.
    """
    if a == b:
        return 1
    else:
        return 0


def q(a, b, theta=2):
    """Binary 'q' function

    Formulation has been taken from [https://www.cmpe.boun.edu.tr/~ozgur/papers/617_Paper.pdf]

    Parameters
    ----------
    a: str
    b: str
    theta: float
        Optional

    Returns
    -------
    theta if a equals b
    1 otherwise
    """
    if a == b:
        return theta
    else:
        return 1


def sim(bigram1, bigram2):
    """Similarity function between bigrams
    
    Formulation has been taken from [https://www.cmpe.boun.edu.tr/~ozgur/papers/617_Paper.pdf]

    Parameters
    ----------
    bigram1: list
        List of bigrams
    
    bigram2: list
        List of bigrams


    Returns
    -------
    float
        Score for 2 bigrams
    """
    dep_1, type_1, head_1 = bigram1
    dep_2, type_2, head_2 = bigram2
    
    return (s(dep_1, dep_2) + s(head_1, head_2)) * q(type_1, type_2)


def check_word_in_question(word, question):
    """Check Word
    
    Checks whether given word is in question or not

    Parameters
    ----------
    word: stanfordnlp.pipeline.doc.Word
        Word to check

    question: str
        Question

    Returns
    -------
    True if word is included
    False otw.
    """
    return word.text.casefold() in question.casefold() 


def extract_answer_from_whole_sent(sentence, question):
    # TODO: Doc-String
    return " ".join([word.text.casefold() for word in sentence.words if word.text.casefold() not in question.casefold() and word.upos != "PUNCT"])


def extract_answer_with_idx(sentence, idx, question):
    # TODO: Doc-String
    return " ".join([word.text.casefold() for word in sentence.words if (word.text.casefold() not in question.casefold() or word.dependency_relation == "nsubj") and int(word.index) in idx and word.upos != "PUNCT"])


def reconstruct_sentence(sentence):
    # TODO: Doc-String
    return " ".join([word.text for word in sentence.words])


def calculate_tree_similarity(sentence, parsed_question):
    """Calculate Dependency Tree Similarity

    Formulation has been taken from [https://www.cmpe.boun.edu.tr/~ozgur/papers/617_Paper.pdf]

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains answer

    parsed_question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    float
        Similarity between dependency trees of sentence and question
    """
    sent_bigrams = construct_bigrams(sentence.words)
    quest_bigrams = construct_bigrams(parsed_question.sentences[0].words)

    score = 0
    for sent_bigram in sent_bigrams:
        for quest_bigram in quest_bigrams:
            score += sim(sent_bigram, quest_bigram)

    return score / (len(sent_bigrams) + len(quest_bigrams))


def calculate_focus_score(sentence, question_focus):
    """Calculate Focus Words Score


    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that contains answer
    
    question_focus: list
        List of focused words

    Returns 
    -------
    float
        Similarity score based on focused words of questionf
    """
    score = 0
    sentence_lemmas = [word.lemma for word in sentence.words]
    for lemma in question_focus:
        if lemma in sentence_lemmas:
            score += 1

    return score / len(question_focus) if question_focus else 0


def calculate_overall_scores(tree_similarities, focus_scores):
    """Combining Tree Similarity and Focus Score


    Parameters
    ----------
    tree_similarities: list
        Dependency tree similarities between question and sentences of passage
    
    focus_score: list
        Focus word similarityies between question and sentences of passage

    Returns
    -------
    list
        Weighted averaged scores
    """
    return [0.8 * tree_dist + 0.2 * focus_score for tree_dist, focus_score in zip(tree_similarities, focus_scores)]


def construct_sentence(sentence, question):
    """Construct Answer
    
    Constructs answer from sentence

    Parameters
    ----------
    sentence: stanfordnlp.pipeline.doc.Sentence
        Sentence that has the answer
    
    question: stanfordnlp.pipeline.doc.Document
        Question

    Returns
    -------
    str
        Answer
    """
    return " ".join([word.text for word in sentence.words if not check_word_in_question(word, question) and word.upos != "PUNCT"])
