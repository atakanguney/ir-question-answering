import copy

from .util import (find_relation,
                   find_root,
                   find_related_words,
                   find_child,
                   construct_answer_from_idx,
                   find_subject)

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


Location = [
    "hangi bölge",
    "hangi il",
    "hangi şehir",
    "hangi şehri",
    "hangi şehre",
    "hangi ülke",
    "hangi kıta",
    "hangi yön",
    "hangi kıta",
    "hangi kent",
    "ülke hangisidir",
    "ülkeler hangileridir",
    "ülkesi hangisidir",
    "şehir hangisidir",
    "şehri hangisidir",
    "şehrimiz hangisidir",
    "ili hangisidir",
    "bölge hangisidir",
    "adası hangisidir",
    "kent hangisidir",
    "neresi",
    "nerelerde",
    "nerede",
    "nereden",
    "nereye",
]

Temporal = [
    "hangi tarihte",
    "hangi çağ",
    "ne zaman",
    "kaçıncı yüzyıl",
    "kaç yılında",
    "dönem hangisidir",
    "hangi dönem",
    "hangi yıl",
    "hangi zaman",
    "hangi jeolojik zaman",
    "hangi sene",
    "dönem hangisidir",
]

Description = [
    "ne denmektedir",
    "ne denir",
    "ne isim verilir",
    "nedir",
    "ne ad verilir",
    "sebebi nedir",
    "hangi ad",
    "hangi faaliyetler",
    "hangi özelli",
    "hangi yönüyle",
    "nasıl",
    " neden ",
    "denir",
]

Entity = [
    "hangi element",
    "hangi antlaşma",
    "hangi anlaşma",
    "hangi kuşak",
    "hangi kuşağ",
    "hangi tip",
    "hangi tür",
    "hangi iklim",
    "hangi maden",
    "hangi kayaç tipi",
    "hangi cevher",
    "hangi şekil",
    "hangi yönetim",
    "hangi sektör",
    "hangi renk",
    "hangi levha",
    "hangi kanun",
    "hangi fay",
    "hangi dağ",
    "hangi biyom",
    "rüzgar tipi",
    "iklim çeşidi",
    "iklim elemanı",
    "havalimanı hangisidir",
    "türü hangisidir",
    "çeşidi hangisidir",
    "şekil hangisidir",
    "ürünü hangisidir",
    "element nedir",
]

Numeric = [
    "ne kadar",
    "kaç",
]

Human = [
    "kim",
    "hangi kurum",
    "hangi kral",
    "hangi kuruluş",
    "hangi padişah",
    "kim",
    "kimin",
    "kimler",
]

YesNo = [
    " mı",
    " mi",
]

proper_classes = {
    "Location" : Location,
    "Temporal": Temporal,
    "Human": Human,
    "Numeric": Numeric,
    "Description": Description,
    "Entity": Entity,
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
    
    
    if "hangi" in q:
        return "Entity"
    elif "ne" in q:
        return "Description"
    elif " mi" in q or " mı" in q:
        return "YesNo"

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
    question_text = question.text.lower()
    question_sent = question.sentences[0]
    if "nedir" in question_text:
        def check_dep(word):
            return word.dependency_relation in ["amod", "nmod:poss"]
        root = find_root(question_sent)

        if question_sent.words[root - 1].text.lower() == "ned":
            
            subj = find_relation(question_sent, root, "nsubj")
            if subj:
                idx = find_related_words(question_sent, subj)
                idx.sort()
                return " ".join([question_sent.words[id_ - 1].text for id_ in idx if check_dep(question_sent.words[id_ - 1]) or question_sent.words[id_ - 1].dependency_relation == "nsubj"])
            obj = find_relation(question_sent, root, "obj")
            if obj:
                idx = find_related_words(question_sent, obj)
                idx.sort()
                return " ".join([question_sent.words[id_ - 1].text for id_ in idx if check_dep(question_sent.words[id_ - 1]) or question_sent.words[id_ - 1].dependency_relation == "obj"])

                
        aux = find_child(question_sent, root, "aux:q")
        if aux > 0 and question_sent.words[aux - 1].text == "nedir":
            return " ".join([word.text for word in question_sent.words if check_dep(word) or word.governor == 0])
    
    elif "verilir" in question_text:
        root = find_root(question_sent)
        nmod = find_relation(question_sent, root, "nmod")
        idx = []
        for id_ in nmod:
            if "Dat" in question_sent.words[id_ - 1].feats:
                idx.append(id_)
                
        if root not in idx:
            idx.append(root)
            
            return construct_answer_from_idx(question_sent, idx)
        
    elif "hangisidir" in question_text:
        subj = find_subject(question_sent)
        
        def check_dep(word):
            dep_rel = word.dependency_relation
            return "cl" in dep_rel or "poss" in dep_rel
        
        if subj:
            subj_ = copy.deepcopy(subj)
            idx = find_related_words(question_sent, subj)
            idx = [id_ for id_ in idx if check_dep(question_sent.words[id_ - 1])] + subj_
            return construct_answer_from_idx(question_sent, idx)
        root = find_root(question_sent)
        nmodposs = find_relation(question_sent, root, "nmod:poss")
        if nmodposs:
            return construct_answer_from_idx(question_sent, nmodposs)
        
        obj = find_relation(question_sent, root, "obj")
        if obj:
            return construct_answer_from_idx(question_sent, obj)

    elif "hangi" in question_text:
        key_word = -1
        for word in question_sent.words:
            if "hangi" in word.text.lower():
                key_word = int(word.index)
                break
                
        if key_word > 0:
            idx = []
            words = question_sent.words
            cur_word = words[words[key_word - 1].governor - 1]
            idx.append(int(cur_word.index))
            while cur_word.governor > 0:
                cur_word = words[cur_word.governor - 1]
                idx.append(int(cur_word.index))
            
            if idx:
                return construct_answer_from_idx(question_sent, idx)
    
    elif "denir" in question_text:
        root = find_root(question_sent)
        nmod = find_relation(question_sent, root, "nmod")
        if nmod:
            return construct_answer_from_idx(question_sent, nmod)
        amod = find_relation(question_sent, root, "amod")
        if amod:
            return construct_answer_from_idx(question_sent, amod)
        obl = find_relation(question_sent, root, "obl")
        if obl:
            return construct_answer_from_idx(question_sent, obl)
    
    elif "kaç" in question_text:
        key_word = -1
        for word in question_sent.words:
            if "kaç" in word.text.lower():
                key_word = int(word.index)
                break
        
        if key_word > 0:
            idx = []
            words = question_sent.words
            cur_word = words[words[key_word - 1].governor - 1]
            idx.append(int(cur_word.index))
            while cur_word.governor > 0:
                cur_word = words[cur_word.governor - 1]
                idx.append(int(cur_word.index))
            
            if idx:
                return construct_answer_from_idx(question_sent, idx)
            
    elif "kadardır" in question_text:
        root = find_root(question_sent)
        cop = find_relation(question_sent, root, "cop")
        punct = find_relation(question_sent, root, "punct")
        
        root = [root] + cop + punct
        subj = find_relation(question_sent, root, "nsubj")

        if subj:
            return construct_answer_from_idx(question_sent, subj)
    
    else:
        root = find_root(question_sent)
        cop = find_relation(question_sent, root, "cop")
        punct = find_relation(question_sent, root, "punct")
        root = [root] + cop + punct
        
        subj = find_relation(question_sent, root, "nsubj")
        poss = find_relation(question_sent, subj, "nmod:poss")
        acl = find_relation(question_sent, subj, "acl")
        advcl = find_relation(question_sent, subj, "advcl")
        
        idx = subj + poss + acl + advcl
        if idx:
            return construct_answer_from_idx(question_sent, idx)

    return " ".join([word.text for word in question_sent.words if word.dependency_relation != "punct"])
