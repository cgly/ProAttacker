import math
import os

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader
import torch
hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class UniversalSentenceEncoder():
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="cosine", **kwargs):
        # super().__init__(threshold=threshold, metric=metric, **kwargs)
        # if large:
        #     tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        # else:
        #     #tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        tfhub_url =r"E:\code\textcode\MyTextAttack\tmp\063d866c06683311b44b4992fd46003be952409c"
        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = hub.load(self._tfhub_url)

        if metric == "cosine":
            self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        elif metric == "angular":
            self.sim_metric = get_angular_sim
        elif metric == "max_euclidean":
            # If the threshold requires embedding similarity measurement
            # be less than or equal to a certain value, just negate it,
            # so that we can still compare to the threshold using >=.
            self.threshold = -threshold
            self.sim_metric = get_neg_euclidean_dist

    def encode(self, sentences):
        # if not self.model:
        #     self.model = hub.load(self._tfhub_url)
        return self.model(sentences).numpy()


    def sim_score(self, starting_text, transformed_text):

        starting_embedding, transformed_embedding = self.encode(
            [starting_text, transformed_text]
        )

        if not isinstance(starting_embedding, torch.Tensor):
            starting_embedding = torch.tensor(starting_embedding)

        if not isinstance(transformed_embedding, torch.Tensor):
            transformed_embedding = torch.tensor(transformed_embedding)

        starting_embedding = torch.unsqueeze(starting_embedding, dim=0)
        transformed_embedding = torch.unsqueeze(transformed_embedding, dim=0)

        return self.sim_metric(starting_embedding, transformed_embedding)

def get_angular_sim(emb1, emb2):
    """Returns the _angular_ similarity between a batch of vector and a batch
    of vectors."""
    cos_sim = torch.nn.CosineSimilarity(dim=1)(emb1, emb2)
    return 1 - (torch.acos(cos_sim) / math.pi)

def get_neg_euclidean_dist(emb1, emb2):
    """Returns the Euclidean distance between a batch of vectors and a batch of
    vectors."""
    return -torch.sum((emb1 - emb2) ** 2, dim=1)

if __name__ == '__main__':
    use = UniversalSentenceEncoder()

    sent1="i love china,I hate japen"
    sent2="i like china,I don't like japen"
    print(use.sim_score(sent1,sent2))