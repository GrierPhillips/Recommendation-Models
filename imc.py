"""
Implementation of Inductive Matric Completion.

Inductive Matrix Completion (IMC) is an algorithm for recommender systems with
side-information of users and items. The IMC formulation incorporates features
associated with rows (users) and columns (items) in matrix completion, so that
it enables predictions for users or items that were not seen during training,
and for which only features are known but no dyadic information (such as
ratings or linkages).
"""


class IMC(object):
    """Implementation of Inductive Matrix Completion."""

    def __init__(self, rank=None, max_iter=30, lambda_=0.1):
        """Create instance of IMC with given parameters."""
        self.rank = rank
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def fit(self):
        pass

    def predict_one(self):
        pass

    def predict_all(self):
        pass

    def score(self):
        pass
