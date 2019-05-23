Entity = [
    "hangisidir",
    "nedir",
    "ne denir",
    "ne ad verilir",
    "nedir",
    "denir",
]

Location = [
    "hangi bölge",
    "hangi il",
    "hangi şehir",
    "hangi ülke",
    "nere",
]

Human = [
    "kim",
]

Numeric = [
    "ne kadar",
    "kaç",
    "hangi tarih",
    "hangi yıl",
    "hangi dönem",
    "hangi çağ",
]

Reason = [
    "nasıl",
]


proper_classes = {
    "Entity": Entity,
    "Location" : Location,
    "Human": Human,
    "Numeric": Numeric,
    "Reason": Reason,
}

def find_q_type(question):
    """Find Question Type

    Parameters
    ----------
    question: str
        Question

    Returns
    -------
    str
        Class Type
    """
    q = question.casefold()
    for class_name, _class in proper_classes.items():
        for key_word in _class:
            if key_word in q:
                return class_name
    
    
    if "ne" in q:
        return "Entity"
    elif " mi" in q or " mı" in q:
        return "YesNo"
    elif "hangi" in q:
        return "Entity"

    return "Other"


def find_q_focus(question):
    """Find Question Focus

    Parameters
    ----------
    question: stanfordnlp.pipeline.doc.Document

    Returns
    -------
    list
        List of words that question focuses
    """

    focus = []
    root_idx = -1
    for word in question.sentences[0].words:
        if word.dependency_relation == "root":
            root_idx = int(word.index)
            focus.append(word.lemma)
            
    def check_word(word):
        return word.dependency_relation != "cop" and word.dependency_relation != "punct"

    for word in question.sentences[0].words:
        if word.governor == root_idx and check_word(word):
            focus.append(word.lemma)
    return focus
