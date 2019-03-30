import numpy as np

from recommenders.RNNRecommender import RNNRecommender
from util import evaluation
from util.data_utils import create_seq_db_filter_top_k
from util.metrics import precision, recall, mrr
from util.split import last_session_out_split


def get_test_sequences(test_data, given_k):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    test_sequences = test_data.loc[test_data['sequence'].map(len) > abs(given_k), 'sequence'].values
    return test_sequences


# dataset_path = 'datasets/sessions.csv'
# load this sample if you experience a severe slowdown with the previous dataset
dataset_path = 'datasets/sessions_sample_10.csv'

# for the sake of speed, let's keep only the top-1k most popular items in the last month
dataset = create_seq_db_filter_top_k(path=dataset_path, topk=1000, last_months=1)

from collections import Counter

cnt = Counter()
dataset.sequence.map(cnt.update);

sequence_length = dataset.sequence.map(len).values
n_sessions_per_user = dataset.groupby('user_id').size()

print('Number of items: {}'.format(len(cnt)))
print('Number of users: {}'.format(dataset.user_id.nunique()))
print('Number of sessions: {}'.format(len(dataset)))

print('\nSession length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    sequence_length.mean(),
    np.quantile(sequence_length, 0.5),
    sequence_length.min(),
    sequence_length.max()))

print('Sessions per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
    n_sessions_per_user.mean(),
    np.quantile(n_sessions_per_user, 0.5),
    n_sessions_per_user.min(),
    n_sessions_per_user.max()))

print('Most popular items: {}'.format(cnt.most_common(5)))

train_data, test_data = last_session_out_split(dataset)
print("Train sessions: {} - Test sessions: {}".format(len(train_data), len(test_data)))

recommender = RNNRecommender(session_layers=[20],
                             batch_size=16,
                             learning_rate=0.1,
                             momentum=0.1,
                             dropout=0.1,
                             epochs=5,
                             personalized=False)
recommender.fit(train_data)
METRICS = {'precision': precision,
           'recall': recall,
           'mrr': mrr}
TOPN = 10  # length of the recommendation list

# GIVEN_K=1, LOOK_AHEAD=1, STEP=1 corresponds to the classical next-item evaluation
GIVEN_K = 1
LOOK_AHEAD = 1
STEP = 1

test_sequences = get_test_sequences(test_data, GIVEN_K)
print('{} sequences available for evaluation'.format(len(test_sequences)))

results = evaluation.sequential_evaluation(recommender,
                                           test_sequences=test_sequences,
                                           given_k=GIVEN_K,
                                           look_ahead=LOOK_AHEAD,
                                           evaluation_functions=METRICS.values(),
                                           top_n=TOPN,
                                           scroll=True,  # scrolling averages metrics over all profile lengths
                                           step=STEP)

print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))
