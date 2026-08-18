import numpy as np
from pathlib import Path
import re
from collections import Counter

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "SMSSpamCollection"
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "not",
    "no",
    "so",
    "too",
    "very",
    "just",
    "get",
    "got",
    "go",
    "going",
}


class dataClass:
    def __init__(self) -> None:
        pass

    def tokenize(self, doc):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", doc.lower())

    def getMsgLabel(self, path=DATA_PATH):
        with open(path, "r", encoding="utf-8") as file:
            Lines = [line.strip() for line in file.readlines()]
        self.messages, self.labels = [], []
        for line in Lines:
            label, _, msg = line.partition("\t")
            self.messages.append(self.tokenize(msg))
            self.labels.append(label)
        return self.messages, self.labels

    def getNi(self, doc):
        Ni = [len(sentence) for sentence in doc]
        return np.array(Ni)

    def getCounts(self, doc):
        return list(Counter(doc).items())

    def buildVocab(self, documents, stopwords=STOPWORDS):
        vocab = set()
        doc_freq = Counter()

        for doc in documents:
            unique = set(doc)
            vocab.update(unique)
            for word in unique:
                doc_freq[word] += 1

        return sorted(w for w in vocab if w not in stopwords), list(doc_freq.items())

    def docs_to_matrix(self, messages, vocab):
        word_indices = {word: idx for idx, word in enumerate(vocab)}
        X = np.zeros((len(messages), len(vocab)), dtype=int)
        row, col, vals = [], [], []
        for idx, msg in enumerate(messages):
            words, counts = np.unique(msg, return_counts=True)
            kept = [(w, c) for w, c in zip(words, counts) if w in word_indices]
            if not kept:
                continue
            kept_words, kept_counts = zip(*kept)
            row.append(np.full(len(kept_words), idx))
            col.append([word_indices[w] for w in kept_words])
            vals.append(np.asarray(kept_counts))
        if row:
            np.add.at(
                X, (np.concatenate(row), np.concatenate(col)), np.concatenate(vals)
            )
        return X

    def labels_to_vector(self, labels):
        return np.array([1 if label == "spam" else 0 for label in labels], dtype=int)

    def split_indices(self, y, test_frac=0.2, seed=42):
        rng = np.random.default_rng(seed)
        train_idx, test_idx = [], []
        for c in np.unique(y):
            ids = np.where(y == c)[0]
            rng.shuffle(ids)
            cut = int(len(ids) * (1 - test_frac))
            train_idx.extend(ids[:cut])
            test_idx.extend(ids[cut:])
        return np.array(train_idx), np.array(test_idx)

    def train_test_split(self, X, y, test_frac=0.2, seed=42):
        train_idx, test_idx = self.split_indices(y, test_frac, seed)
        return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])

    @staticmethod
    def getLength(element):
        return len(element)


if __name__ == "__main__":
    data = dataClass()
    messages, labels = data.getMsgLabel()
    y = data.labels_to_vector(labels)

    tr_idx, te_idx = data.split_indices(y)
    vocab, _ = data.buildVocab([messages[i] for i in tr_idx])
    X = data.docs_to_matrix(messages, vocab)

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    print(f"train: {X_train.shape}  test: {X_test.shape}  vocab: {len(vocab)}")
    print(
        f"spam in train: {(y_train == 1).sum()}/{len(y_train)}  "
        f"test: {(y_test == 1).sum()}/{len(y_test)}"
    )
