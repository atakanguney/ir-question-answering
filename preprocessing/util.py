class Question(object):
    """Question Class"""
    def __init__(self, idx, text):
        self.idx = idx
        self.text = text


class Answer(object):
    """Answer Class"""
    def __init__(self, idx, text):
        self.idx = idx
        self.text = text


class QuestionGroup(object):
    """Question Group Class"""
    def __init__(self, questions, answer, related_par_id):
        self.questions = questions
        self.answer = answer
        self.related_par_id = related_par_id
