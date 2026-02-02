import zlib

text = "is the set of all possible elements, and the set of all possible elements is the universe. The set of all possible elements is the set of all elements that exist in the universe. The set of all elements that exist in the universe is the set of all possible elements. The set of all possible elements is the set of all elements that exist in the universe. The set of all elements that exist in the universe is the set of all possible elements."

# 1. Current Heuristic Simulation (Trigram)
words = text.split()
trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
print(f"Total Trigrams: {len(trigrams)}")
unique_trigrams = len(set(trigrams))
print(f"Unique Trigrams: {unique_trigrams}")
print(f"Ratio: {unique_trigrams / len(trigrams):.2f}")

# 2. Compression Ratio
compressed = zlib.compress(text.encode('utf-8'))
ratio = len(text) / len(compressed)
print(f"Length: {len(text)}, Compressed: {len(compressed)}")
print(f"Compression Ratio: {ratio:.2f}")

# 3. Last Sentence overlap
sentences = text.split('. ')
print(f"Sentences: {len(sentences)}")
if len(sentences) > 2:
    s1 = set(sentences[-1].split())
    s2 = set(sentences[-2].split())
    overlap = len(s1.intersection(s2)) / len(s1.union(s2))
    print(f"Semantic Overlap: {overlap:.2f}")
