from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

corpus_file = "corpus.txt"
with open(corpus_file, "r", encoding="utf-8") as f:
    corpus_text = f.read()

sentences = []
for line in corpus_text.strip().split('\n'):
    if line.strip():
        # Convert to lowercase and tokenize
        tokens = simple_preprocess(line, deacc=True)
        if tokens:
            sentences.append(tokens)

print(f"Number of sentences: {len(sentences)}")
print(f"Sample sentence: {sentences[0][:10]}...")

configurations = [
    {'vector_size': 100, 'window': 10, 'sg': 1, 'epochs': 200, 'min_count': 1, 'seed': 42},
    {'vector_size': 150, 'window': 8, 'sg': 0, 'epochs': 150, 'min_count': 1, 'seed': 42},
    {'vector_size': 50, 'window': 15, 'sg': 1, 'epochs': 300, 'min_count': 1, 'seed': 123},
]

best_results = []

for i, config in enumerate(configurations):
    print(f"\n{'=' * 60}")
    print(f"CONFIGURATION {i + 1}: {config}")
    print('=' * 60)

    model = Word2Vec(sentences=sentences, workers=4, **config)

    # Solve analogies
    try:
        results1 = model.wv.most_similar(positive=['doctors', 'law'], negative=['medicine'], topn=5)
        results2 = model.wv.most_similar(positive=['teachers', 'hospitals'], negative=['schools'], topn=5)

        word1 = results1[0][0]
        word2 = results2[0][0]

        print(f"Analogy 1 results: {results1}")
        print(f"Analogy 2 results: {results2}")
        print(f"Flag: SIGMOID_{word1.upper()}_{word2.upper()}")

        best_results.append((word1, word2, config))
    except Exception as e:
        print(f"Error: {e}")

# Use the first configuration for final result
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=10,
    min_count=1,
    workers=4,
    sg=1,
    epochs=200,
    seed=42
)

print("\nVocabulary size:", len(model.wv))
print("Key words in vocabulary:",
      [w for w in ['doctors', 'medicine', 'law', 'teachers', 'schools', 'hospitals', 'engineers', 'athletes']])


# Function to solve analogy: A - B + C = ?
def solve_analogy(model, word_a, word_b, word_c, top_n=5):
    """
    Solves analogy: A - B + C = ?
    Returns the most similar word to (C + (A - B))
    """
    print(f"\nSolving: {word_a} - {word_b} + {word_c} = ?")

    try:
        # Use gensim's built-in most_similar method with positive and negative
        results = model.wv.most_similar(
            positive=[word_a, word_c],
            negative=[word_b],
            topn=top_n
        )

        print(f"Top {top_n} results:")
        for word, similarity in results:
            print(f"  {word}: {similarity:.4f}")

        return results[0][0]  # Return the most similar word
    except KeyError as e:
        print(f"Error: Word not found in vocabulary: {e}")
        return None


# Solve the two analogies with detailed output
print("\n" + "=" * 60)
print("SOLVING ANALOGIES - DETAILED ANALYSIS")
print("=" * 60)

# 1: doctors - medicine + law = ?
print("\nAnalogy 1: doctors - medicine + law = ?")
word1_results = solve_analogy(model, 'doctors', 'medicine', 'law', top_n=15)

# 2: teachers - schools + hospitals = ?
print("\nAnalogy 2: teachers - schools + hospitals = ?")
word2_results = solve_analogy(model, 'teachers', 'schools', 'hospitals', top_n=15)

# Try different candidate combinations
print("\n" + "=" * 60)
print("POSSIBLE FLAGS (trying top candidates):")
print("=" * 60)

if word1_results and word2_results:
    word1 = word1_results
    word2 = word2_results

    # Get top results for manual inspection
    results1 = model.wv.most_similar(positive=['doctors', 'law'], negative=['medicine'], topn=5)
    results2 = model.wv.most_similar(positive=['teachers', 'hospitals'], negative=['schools'], topn=5)

    # Try combinations of top 3 from each
    for i, (w1, s1) in enumerate(results1[:3]):
        for j, (w2, s2) in enumerate(results2[:3]):
            flag = f"SIGMOID_{w1.upper()}_{w2.upper()}"
            print(f"{i + 1},{j + 1}: {flag} (similarities: {s1:.3f}, {s2:.3f})")

    # Primary answer
    flag = f"SIGMOID_{word1.upper()}_{word2.upper()}"
    print(f"\n🚩 PRIMARY FLAG: {flag}")
else:
    print("\nError: Could not solve one or both analogies")