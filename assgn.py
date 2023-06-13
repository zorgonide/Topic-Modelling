import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd
import re
stop_words = []

with open('../stopwordFile.txt', "r") as f:
    stop_words = f.read().split()

tweets_df = pd.read_csv('tweets.csv')
tweets = tweets_df.iloc[:, 2]


def preprocess_tweets(text):
    # Convert to list of words
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    # text = re.sub(r'@\w+', '', text)
    words = simple_preprocess(text, deacc=True)
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    return words


if __name__ == '__main__':
    tweets_df['tokens'] = tweets_df['text'].apply(preprocess_tweets)
    # tweets_df['tokens'] = (preprocess_tweets(text) for text in tweets_df['text'])

    dictionary = corpora.Dictionary(tweets_df['tokens'])
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in tweets_df['tokens']]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=10,
                                                random_state=42,
                                                update_every=1,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # Print the topics and their keywords
    topics = lda_model.print_topics()
    for topic in topics:
        print(topic)

    # Compute coherence score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=tweets_df['tokens'], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
