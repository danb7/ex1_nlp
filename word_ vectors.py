import gensim.downloader as dl
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from utils import *


model = dl.load("word2vec-google-news-300")
vocab = model.index_to_key

# Generating lists of the most similar words
print('Generating lists of the most similar words:')
print('_' * len('Generating lists of the most similar words:'))
five_words = ['espresso', 'game', 'spy', 'car', 'smartphone']
generate_most_similar_words(model, five_words, 20)

# Polysemous Words
print('Polysemous Words:')
print('_' * len('Polysemous Words:'))
polysemous_words_group_1 = ['mole', 'bass', 'fall']
polysemous_words_group_2 = ['rock', 'bar', 'plant']
print(indent_level + 'group 1 (neighbors reflect both word meanings):')
generate_most_similar_words(model, polysemous_words_group_1, 10, init_indent=indent_level)
print(indent_level + 'group 2 (neighbors reflect single meanings):')
generate_most_similar_words(model, polysemous_words_group_2, 10, init_indent=indent_level)

# Synonyms and Antonyms
print('Synonyms and Antonyms:')
print('_' * len('Synonyms and Antonyms:'))
w1, w2, w3 = 'love', 'like', 'hate'
sim_w1_w2 = model.similarity(w1,w2)
sim_w1_w3 = model.similarity(w1,w3)
print(indent_level + f'similarity between w1 ({w1}) and w2 ({w2}): {sim_w1_w2}')
print(indent_level + f'similarity between w1 ({w1}) and w3 ({w3}): {sim_w1_w3}')
print(indent_level + 'w1 and w2 are synonyms, w1 and w3 are antonyms and sim(w1,w2) < sim(w1, w3)!')

# The Effect of Different Corpora
print('The Effect of Different Corpora:')
print('_' * len('The Effect of Different Corpora:'))
wiki_model = dl.load("glove-wiki-gigaword-200")
twitter_model = dl.load("glove-twitter-200")
sim_cross_corpus_words = ['yellow', 'morning', 'dog', 'car', 'coffee']
different_cross_corpus_words = ['umbrella', 'troll', 'profile', 'mute', 'gaming']
print(indent_level + ('5 words whose top 10 neighbors based on the news corpus are very similar '
                      'to their top 10 neighbors based on the twitter corpus:'))
compare_models_similarity(wiki_model, twitter_model, sim_cross_corpus_words, 10, indent_level*2)
print(indent_level + ('5 words whose top 10 neighbors based on the news corpus are '
                      'substantially different from the top 10 neighbors based '
                      'on the twitter corpus:'))
compare_models_similarity(wiki_model, twitter_model, different_cross_corpus_words, 10, indent_level*2)

# Dimensionality Reduction
print('Dimensionality Reduction')
print('_' * len('Dimensionality Reduction'))    
first_5000_words = vocab[1:5000]
past_verbs = [word for word in first_5000_words if word.endswith("ed")]
present_verbs = [word for word in first_5000_words if word.endswith("ing")]
all_verbs = past_verbs + present_verbs
past_verbs_idx = range(len(past_verbs))
present_verbs_idx = range(len(past_verbs), len(all_verbs))
labels = ['ed']*len(past_verbs) + ['ing']*len(present_verbs)
print(indent_level + f'sanity check, should be 708. number of words={len(all_verbs)}')
verbs_vectors = [model[word] for word in all_verbs]

pca = decomposition.PCA(n_components=2)
Z = pca.fit_transform(verbs_vectors)

print(indent_level + (f'Percentage of variance explained by the two components: '
                      f'{pca.explained_variance_ratio_}'))
print(indent_level + (f'together they explained just '
                      f'{pca.explained_variance_ratio_.sum()} of the variance!'))

plt.scatter(Z[past_verbs_idx, 0], Z[past_verbs_idx, 1], c='blue', label='ed')
plt.scatter(Z[present_verbs_idx, 0], Z[present_verbs_idx, 1], c='green', label='ing')
plt.legend()
plt.title("2D Scatter Plot after PCA of words ending with 'ed' or 'ing")
plt.savefig('pca_plot.png')
print(indent_level + 'plot saved in .\pca_plot.png')

print('\nfinish!')
# plt.show()

#Word-similarities in Large Language Model
print("Word-similarities in Large Language Model")
two_words = ['run', 'espresso']
print('Generating lists of the 100 most similar words:')
print('_' * len('Generating lists of the 100 most similar words:'))
generate_most_similar_words(model, two_words, 100)