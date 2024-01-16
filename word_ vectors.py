import gensim.downloader as dl
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


model = dl.load("word2vec-google-news-300")
# this will take a while on first load as it downloads a 1.6G file.
# later calls will be cached.
# You can now use various methods of the “model“ object.
# you can access the vocabulary like so:
vocab = model.index_to_key

indent_level = '\t' # to print hierarchical and more clear

# Generating lists of the most similar words
print('Generating lists of the most similar words:')
print('_' * len('Generating lists of the most similar words:'))
five_words = ['espresso', 'game', 'spy', 'car', 'smartphone']

def get_most_similar_words(model, words, n=20):
    most_similar_dict = {}
    for w in words:
        most_similar_dict[w] = model.most_similar(w, topn=n)
    return most_similar_dict

def display_sim(similarity_dict, init_indent=''):
    for w, l in similarity_dict.items():
        print(init_indent+f'{w} - Top {len(l)} most similar words:')
        for w2 in l:
            print(init_indent+indent_level, w2)

most_similar_five_words = get_most_similar_words(model, five_words, 20)
display_sim(most_similar_five_words, indent_level)

# Polysemous Words
print('Polysemous Words:')
print('_' * len('Polysemous Words:'))
polysemous_words = ['paper', 'head', 'plant', 'board', 'rock', 'match', 'pupil',
                    'club', 'bar', 'draft', 'clip', 'second', 'train',
                    'race', 'left', 'story', 'fan', 'conductor', 'draw', 'bank', 'book',
                    'arm', 'spot', 'bass', 'clip', 'key', 'turkey', 'jordan', 'python', 'general',
                    'close', 'string', 'box', 'mouse', 'ruler',
                    'bear', 'mint', 'mole', 'palm', 'fall', 'like']
poly_group_1 = ['mole', 'bass', 'fall']
poly_group_2 = ['rock', 'bar', 'plant']
most_similar_poly_dict = get_most_similar_words(model, polysemous_words, 10)
display_sim(most_similar_poly_dict, indent_level)

# Synonyms and Antonyms
print('Synonyms and Antonyms:')
print('_' * len('Synonyms and Antonyms:'))
w1, w2, w3 = 'happy', 'joyful', 'sad'
sim_w1_w2 = model.similarity(w1,w2)
sim_w1_w3 = model.similarity(w1,w3)
print(indent_level+f'similarity between w1 ({w1}) and w2 ({w2}): {sim_w1_w2}')
print(indent_level+f'similarity between w1 ({w1}) and w3 ({w3}): {sim_w1_w3}')
print(indent_level+'w1 and w2 are synonyms, w1 and w3 are antonyms and sim(w1,w2) < sim(w1, w3)!')

# The Effect of Different Corpora
print('The Effect of Different Corpora:')
print('_' * len('The Effect of Different Corpora:'))
wiki_model = dl.load("glove-wiki-gigaword-200")
twitter_model = dl.load("glove-twitter-200")

def compare_models_similarity(model1, model2, words, n, init_indent=''):
    for w in words:
        sim_model_1 = model1.most_similar(w, topn=n)
        sim_model_2 = model2.most_similar(w, topn=n)
        print(init_indent+f'{w} - Top {n} most similar words by each model:')
        for i in range(n):
            print( init_indent+(f'\t ({sim_model_1[i][0]}, {round(sim_model_1[i][1], 2)})\t|\t'
                                f'({sim_model_2[i][0]}, {round(sim_model_2[i][1], 2)})')
            )


sim_cross_corpus_words = ['number', 'morning', 'dog', 'helicopter', 'coffee']
different_cross_corpus_words = ['umbrella', 'troll', 'profile', 'mute', 'gaming']
compare_models_similarity(wiki_model, twitter_model, sim_cross_corpus_words, 10, indent_level)
compare_models_similarity(wiki_model, twitter_model, different_cross_corpus_words, 10, indent_level)

# Dimensionality Reduction
print('Dimensionality Reduction')
print('_' * len('Dimensionality Reduction'))    
first_5000_words = vocab[1:5000]
past_verbs = [word for word in first_5000_words if word.endswith("ed")]
present_verbs = [word for word in first_5000_words if word.endswith("ing")]
all_verbs = past_verbs + present_verbs
past_verbs_idx = range(len(past_verbs))
present_verbs_idx = range(len(past_verbs), len(all_verbs))
# colors = ['blue']*len(past_verbs) + ['green']*len(present_verbs)
labels = ['ed']*len(past_verbs) + ['ing']*len(present_verbs)
print(indent_level + f'sanity check, should be 708. number of words={len(all_verbs)}')
verbs_vectors = [model[word] for word in all_verbs]

pca = decomposition.PCA(n_components=2)
Z = pca.fit_transform(verbs_vectors)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Z, labels)
# class_probabilities = knn.predict_proba(Z)
preds = knn.predict(Z)

mixed_indices = np.where(labels!=preds)[0]

plt.scatter(Z[past_verbs_idx, 0], Z[past_verbs_idx, 1], c='blue', label='ed')
plt.scatter(Z[present_verbs_idx, 0], Z[present_verbs_idx, 1], c='green', label='ing')

# Annotate controversial points
for i in np.random.choice(mixed_indices, 5, replace=False):
    plt.annotate(f'{all_verbs[i]}', (Z[i, 0], Z[i, 1]),
                     textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='red')
    # also annotate the neighbors
    nbrs = knn.kneighbors(Z[i,:].reshape(1,2))
    for ei, i in enumerate(nbrs[1].flatten()[1:]):
        plt.annotate(f'{all_verbs[i]}', (Z[i, 0], Z[i, 1]),
                     textcoords="offset points", xytext=(-5**ei, -5), ha='center', fontsize=8, color='black')
    
plt.legend()
plt.title("2D Scatter Plot after PCA of words ending with 'ed' or 'ing")
plt.savefig('pca_plot.png')
# plt.show()
