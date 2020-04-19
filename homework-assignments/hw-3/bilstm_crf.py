import torch
import torch.nn as nn
from sklearn import metrics


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, state_size, embedding_dim, feature_dim):
        """Linear-chain CRF with Bi-directional LSTM features.

        Args:
            vocab_size (int): num of word types in dictionary
            state_size (int): num of states
            embedding_dim (int): word embedding dim size
            feature_dim (int): word feature dim size
        """
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # word features
        # obs_seq [seq, batch] -> embed_seq [seq, batch, embed]
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        # embed_seq [seq, batch, embed] -> bi-LSTM feature_seq [seq, batch, dir * feat]
        self.lstm = nn.LSTM(embedding_dim, self.feature_dim, num_layers=1, bidirectional=True)

        # singleton potential
        # feature_seq [seq, batch, dir * feat] -> singleton potential [seq, batch, state]
        self.linear_transform = nn.Linear(2 * self.feature_dim, self.state_size)

        # pairwise potential
        # [state at time t-1, state at time t], independent of obs
        self.pairwise_score = nn.Parameter(torch.randn(self.state_size, self.state_size))
        self.start_score = nn.Parameter(torch.randn(self.state_size))
        self.stop_score = nn.Parameter(torch.randn(self.state_size))

    def lstm_features(self, obs_seq):
        """Generate features for each observation in the sequence using LSTM.

        Args:
            obs_seq (torch.LongTensor): [seq, batch]

        Returns:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]
        """
        seq, batch = obs_seq.shape
        embed_seq = self.embedding(obs_seq)  # [seq, batch, embed]
        h_in = torch.randn(2, batch, self.feature_dim)  # [layer * dir, batch, feat]
        c_in = torch.randn(2, batch, self.feature_dim)  # [layer * dir, batch, feat]
        feature_seq, _ = self.lstm(embed_seq, (h_in, c_in))  # [seq, batch, dir * feat]
        return feature_seq

    def log_partition(self, feature_seq):
        """Compute log partition function log Z(x_{1:T}) for features phi_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]

        Returns:
            log_partition (torch.FloatTensor): [batch]
        """
        # TODO: Implement this
        return log_partition

    def log_score(self, feature_seq, state_seq):
        """Compute log score p_tilde(z_{1:T}, x_{1:T}) for features phi_{1:T} and states z_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]
            state_seq (torch.LongTensor): [seq, batch]

        Returns:
            log_score (torch.FloatTensor): [batch]
        """
        seq, batch, _ = feature_seq.shape
        singleton_score = self.linear_transform(feature_seq)  # [seq, batch, state]
        score = singleton_score[0, torch.arange(batch), state_seq[0]]  # [batch]
        score += self.start_score[state_seq[0]]  # [batch]
        for t in range(1, seq):
            score += singleton_score[t, torch.arange(batch), state_seq[t]]  # [batch]
            score += self.pairwise_score[state_seq[t-1], state_seq[t]]  # [batch]
        score += self.stop_score[state_seq[-1]]  # [batch]
        return score

    def neg_log_likelihood(self, obs_seq, state_seq):
        """Compute negative log-likelihood for observation x_{1:T} and states z_{1:T}.

        Args:
            obs_seq (torch.LongTensor): [seq, batch]
            state_seq (torch.LongTensor): [seq, batch]

        Returns:
            nll (float): negative log likelihood of the input
        """
        feature_seq = self.lstm_features(obs_seq)  # [seq, batch, dir * feat]
        log_partition = self.log_partition(feature_seq)  # [batch]
        log_score = self.log_score(feature_seq, state_seq)  # [batch]
        nll = torch.mean(log_partition - log_score)  # float
        return nll

    def viterbi_decode(self, feature_seq):
        """Viterbi decoding for features phi_{1:T}.

        Args:
            feature_seq (torch.FloatTensor): [seq, batch, dir * feat]

        Returns:
            max_score (torch.FloatTensor): [batch]
            state_seq (torch.LongTensor): [seq, batch]
        """
        # TODO: Implement this
        return max_score, state_seq

    def forward(self, obs_seq):
        """

        Args:
            obs_seq (torch.LongTensor): [seq, batch]

        Returns:
            max_score (torch.FloatTensor): [batch]
            state_seq (torch.LongTensor): [seq, batch]
        """
        feature_seq = self.lstm_features(obs_seq)  # [seq, batch, dir * feat]
        max_score, state_seq = self.viterbi_decode(feature_seq)  # [batch], [seq, batch]
        return max_score, state_seq


def load_iob2(filename):
    """Load IOB2 format file as a list of sentences and tags.

    data = [(sentence1, tags1), (sentence2, tags2), ...]
    sentence = [word1, word2, ...]
    tags = [tag1, tags, ...]

    Args:
        filename (str): IOB2 format file

    Returns:
        data (list): list of sencence and tags
    """
    data = []
    with open(filename, 'r') as f:
        sentence, tags = [], []
        for line in f:
            if line.isspace():
                data.append((sentence, tags))
                sentence, tags = [], []
            else:
                tag, word = line.split()
                sentence.append(word)
                tags.append(tag)
    return data


def main():
    torch.manual_seed(123)

    # Toy data
    training_data = [
        ("Peter    Piper    picked a peck of pickled peppers".split(),
         "B-Person I-Person O      O O    O  B-Food  I-Food ".split()),
        ("A peck of pickled peppers Peter    Piper    picked".split(),
         "O O    O  B-Food  I-Food  B-Person I-Person O".split())
    ]

    # If toy data is boring, here's a real dataset to play with.
    # https://github.com/juand-r/entity-recognition-datasets/tree/master/data/MITRestaurantCorpus
    # https://groups.csail.mit.edu/sls/downloads/restaurant/
    # training_data = load_iob2('restauranttrain.bio')
    # testing_data = load_iob2('restauranttest.bio')

    # build dictionary for words and tags
    word2idx, tag2idx = {}, {}
    for sentence, tags in training_data:
        for word, tag in zip(sentence, tags):
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    # Model and optimizer
    model = BiLSTM_CRF(len(word2idx), len(tag2idx), embedding_dim=5, feature_dim=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Train entire model
    for epoch in range(300):
        for sentence, tags in training_data:
            optimizer.zero_grad()
            # Padding is tedious, so let's keep batch=1.
            sentence_idx = torch.tensor([word2idx[word] for word in sentence], dtype=torch.long).view(-1, 1)
            tags_idx = torch.tensor([tag2idx[tag] for tag in tags], dtype=torch.long).view(-1, 1)
            loss = model.neg_log_likelihood(sentence_idx, tags_idx)
            # print('epoch', epoch, 'loss', loss.item())
            loss.backward()
            optimizer.step()

    # Print predictions
    with torch.no_grad():
        y_true, y_pred = [], []
        for sentence, tags in training_data:
            sentence_idx = torch.tensor([word2idx[word] for word in sentence], dtype=torch.long).view(-1, 1)
            max_score, tags_pred = model(sentence_idx)
            tags_pred = [idx2tag[idx] for idx in tags_pred[:, 0].tolist()]
            print(sentence)
            print(tags_pred)
            print(max_score)
            y_true += tags
            y_pred += tags_pred
        print(metrics.classification_report(y_true=y_true, y_pred=y_pred))


if __name__ == '__main__':
    main()
