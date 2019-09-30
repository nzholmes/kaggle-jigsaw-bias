from pathlib import Path

TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_TOXICITY_COLUMNS = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
Y_COLUMNS = [TOXICITY_COLUMN, TOXICITY_COLUMN]+AUX_TOXICITY_COLUMNS

EMBEDDING_CRAWL = '../../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
EMBEDDING_GLOVE = '../../input/glove840b300dtxt/glove.840B.300d.txt'

DATA_DIR = Path('../../input/jigsaw-unintended-bias-in-toxicity-classification/')
TRAIN_DATA = DATA_DIR / 'train.csv'
TEST_DATA = DATA_DIR / 'test.csv'

OUTPUT_DIR = Path('../../output/')
TRAIN_PREPROCESSED_DATA = DATA_DIR / 'train_preprocessed.csv'
TEST_PREPROCESSED_DATA = DATA_DIR / 'test_preprocessed.csv'
