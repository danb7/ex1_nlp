
indent_level = '\t' # to print hierarchical and more clear


def get_most_similar_words(model, words, n=20):
    """
    returns dict: The top-n most similar words to each one in words, by the given model.
    """
    most_similar_dict = {}
    for w in words:
        most_similar_dict[w] = model.most_similar(w, topn=n)
    return most_similar_dict

def display_sim(similarity_dict, init_indent=''):
    """
    print more nicely.
    """
    for w, l in similarity_dict.items():
        print(init_indent+f'{w} - Top {len(l)} most similar words:')
        for w2 in l:
            print(init_indent+indent_level, w2)

def generate_most_similar_words(model, words, n=20, init_indent=''):
    """
    Generating lists of the most similar words
    """
    most_similar_words = get_most_similar_words(model, words, n)
    display_sim(most_similar_words, init_indent+indent_level)


def compare_models_similarity(model1, model2, words, n, init_indent=''):
    """
    display top-n similar word by each model side by side
    """
    for w in words:
        sim_model_1 = model1.most_similar(w, topn=n)
        sim_model_2 = model2.most_similar(w, topn=n)
        print(init_indent+f'{w} - Top {n} most similar words by each model:')
        for i in range(n):
            print( init_indent+(f'\t ({sim_model_1[i][0]}, {round(sim_model_1[i][1], 2)})\t|\t'
                                f'({sim_model_2[i][0]}, {round(sim_model_2[i][1], 2)})')
            )
