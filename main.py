# POS Tagging using Hidden Markov Model
import nltk
from nltk.corpus import treebank
import numpy as np
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('punkt_tab')   # <-- add this new line


nltk.download('treebank')
nltk.download('universal_tagset')

# Load dataset
data = treebank.tagged_sents(tagset='universal')
train_data = data[:3500]
test_data = data[3500:]

# Extract states (tags) and observations (words)
tags = set(tag for sent in train_data for (word, tag) in sent)
words = set(word.lower() for sent in train_data for (word, tag) in sent)

# Initialize probability dictionaries
transition_probs = {}
emission_probs = {}
tag_counts = {}

# Calculate transition and emission probabilities
for sent in train_data:
    prev_tag = "<s>"
    for word, tag in sent:
        word = word.lower()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        emission_probs[(tag, word)] = emission_probs.get((tag, word), 0) + 1
        transition_probs[(prev_tag, tag)] = transition_probs.get((prev_tag, tag), 0) + 1
        prev_tag = tag
    transition_probs[(prev_tag, "</s>")] = transition_probs.get((prev_tag, "</s>"), 0) + 1

# Normalize
for (prev_tag, tag) in transition_probs:
    total = sum(v for (pt, t), v in transition_probs.items() if pt == prev_tag)
    transition_probs[(prev_tag, tag)] /= total

for (tag, word) in emission_probs:
    total = tag_counts[tag]
    emission_probs[(tag, word)] /= total

# Viterbi algorithm
def viterbi(sentence):
    sentence = [w.lower() for w in nltk.word_tokenize(sentence)]
    V = [{}]
    path = {}
    for tag in tags:
        V[0][tag] = transition_probs.get(("<s>", tag), 1e-6) * emission_probs.get((tag, sentence[0]), 1e-6)
        path[tag] = [tag]
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        for tag in tags:
            (prob, state) = max((V[t-1][y0] * transition_probs.get((y0, tag), 1e-6) *
                                 emission_probs.get((tag, sentence[t]), 1e-6), y0) for y0 in tags)
            V[t][tag] = prob
            new_path[tag] = path[state] + [tag]
        path = new_path
    (prob, state) = max((V[len(sentence)-1][tag] * transition_probs.get((tag, "</s>"), 1e-6), tag) for tag in tags)
    return list(zip(sentence, path[state]))

# Test
sentence = "Time flies like an arrow"
result = viterbi(sentence)
print("\nTagged Sentence:\n", result)

# Visualization
tags_list = [tag for (_, tag) in result]
plt.figure(figsize=(8,3))
plt.plot(tags_list, marker='o')
plt.title("POS Tag Sequence (HMM)")
plt.xlabel("Word Index")
plt.ylabel("Tag")
plt.show()
