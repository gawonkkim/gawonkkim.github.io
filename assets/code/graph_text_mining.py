import pandas as pd
import numpy as np
import nltk
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

text = """Coffee is one of the most beloved beverages worldwide, enjoyed by millions of people every day.
The history of coffee dates back to the 15th century when it was first discovered in Ethiopia. From there, it spread to the Arabian Peninsula and
eventually to Europe and the Americas. Coffee is made from roasted coffee beans, which are the seeds of berries from the Coffea plant.
The two most common types of coffee beans are Arabica and Robusta, each offering distinct flavors and aromas. Coffee not only provides a caffeine boost
but is also rich in antioxidants, which can help protect against various diseases. The coffee industry is a major economic driver, with countries like Brazil,
Vietnam, and Colombia being the top producers. Coffee culture has evolved significantly, with numerous brewing methods and a variety of specialty coffee drinks available today.
"""

# Stop words identification
stop_words = set(stopwords.words('english'))

# Tokenization
tokens = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]

# Part-of-Speech Tagging
pos_tags = nltk.pos_tag(tokens)

# Token Filtering for Nouns and Adjectives
filtered_tokens = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('JJ')]

def keyword_extraction_textrank(tokens, top_n=10):
    word_graph = nx.Graph()

    for i, word1 in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            word2 = tokens[j]
            if word1 != word2:
                if word_graph.has_edge(word1, word2):
                    word_graph[word1][word2]['weight'] += 1
                else:
                    word_graph.add_edge(word1, word2, weight=1)

    ranks = nx.pagerank(word_graph)
    sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
    keyphrases = [word for word, rank in sorted_ranks[:top_n]]
    return keyphrases

keyphrases = keyword_extraction_textrank(filtered_tokens)
print("TextRank Key Phrases:", keyphrases)

print("Tokenization: ", tokens)
print("Part-of-Speach Tagging:", pos_tags)

def sentence_similarity(sent1, sent2):
    sent1 = [word for word in word_tokenize(sent1.lower()) if word.isalpha() and word not in stop_words]
    sent2 = [word for word in word_tokenize(sent2.lower()) if word.isalpha() and word not in stop_words]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        vector1[all_words.index(word)] += 1

    for word in sent2:
        vector2[all_words.index(word)] += 1

    return cosine_similarity([vector1], [vector2])[0][0]

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    return similarity_matrix

def summarize_text_textrank(text, top_n=3):
    sentences = sent_tokenize(text)
    similarity_matrix = build_similarity_matrix(sentences)

    sentence_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = [ranked_sentences[i][1] for i in range(top_n)]
    return " ".join(summary)

summary = summarize_text_textrank(text)
print("TextRank Summary:", summary)
