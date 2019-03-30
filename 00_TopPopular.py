from collections import Counter

from recommenders.PopularityRecommender import PopularityRecommender
from util import evaluation
from util.data_utils import create_seq_db_filter_top_k
from util.metrics import precision, recall, mrr, ndcg
from util.split import last_session_out_split


def get_test_sequences(test_data, given_k):
    # we can run evaluation only over sequences longer than abs(LAST_K)
    test_sequences = test_data.loc[test_data['sequence'].map(len) > abs(given_k), 'sequence'].values
    return test_sequences


# dataset_path = 'datasets/sessions.csv'
# load this sample if you experience a severe slowdown with the previous dataset
# dataset_path = 'datasets/sessions_sample_10.csv'
dataset_path = 'C:\\Users\\Hailin\\Documents\\Projects\\Dataset\\MovieLens\\ml-1m\\ratings.dat'

# for the sake of speed, let's keep only the top-1k most popular items in the last month
dataset = create_seq_db_filter_top_k(path=dataset_path, topk=100)

cnt = Counter()
dataset['sequence'].map(cnt.update);

sequence_length = dataset['sequence'].map(len).values

print('Number of users: {}'.format(len(dataset)))
# print('Number of items: {}'.format(dataset['sequence'].nunique()))

print('Most popular items: {}'.format(cnt.most_common(5)))

train_data, test_data = last_session_out_split(dataset, user_key='UserID', session_key='sequence', time_key='Timestamp')
print("Train sessions: {} - Test sessions: {}".format(len(train_data), len(test_data)))

recommender = PopularityRecommender()
recommender.fit(train_data)

METRICS = {'precision': precision,
           'recall': recall,
           'mrr': mrr,
           'ndcg': ndcg}
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

GIVEN_K = 1
LOOK_AHEAD = 'all'
STEP = 1

test_sequences = get_test_sequences(test_data, GIVEN_K)
print('{} sequences available for evaluation'.format(len(test_sequences)))

results = evaluation.sequential_evaluation(recommender,
                                           test_sequences=test_sequences,
                                           given_k=GIVEN_K,
                                           look_ahead=LOOK_AHEAD,
                                           evaluation_functions=METRICS.values(),
                                           top_n=TOPN,
                                           scroll=False  # notice that scrolling is disabled!
                                           )

print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
for mname, mvalue in zip(METRICS.keys(), results):
    print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))
