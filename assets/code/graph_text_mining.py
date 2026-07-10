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

text = """Artificial intelligence has become one of the most transformative technologies of the 21st century, reshaping industries from finance to healthcare.
Machine learning, a core branch of AI, enables systems to learn patterns from data rather than following explicitly programmed rules. Deep learning, which relies on neural networks
with many layers, has driven major breakthroughs in image recognition, natural language processing, and game-playing agents. As AI models grow larger and more capable, questions
about interpretability, fairness, and responsible deployment have become increasingly important. Economists have also started studying how AI adoption affects productivity, labor markets,
and competition across industries. Understanding both the technical foundations and the broader economic implications of AI is essential for building systems that are not only powerful but also trustworthy.
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
