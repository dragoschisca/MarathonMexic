import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_corpus(filename='corpus.txt'):
    """Load and preprocess the corpus file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize sentences
        sentences = []
        for line in text.split('\n'):
            if line.strip():
                # Convert to lowercase and tokenize
                tokens = simple_preprocess(line, deacc=True)
                if tokens:
                    sentences.append(tokens)

        print(f"Loaded {len(sentences)} sentences from corpus")
        return sentences

    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None


def train_word2vec(sentences, vector_size=100, window=5, min_count=1, epochs=100):
    """Train Word2Vec model on the corpus."""
    print("\nTraining Word2Vec model...")

    # Train with multiple parameters for better results
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-gram model (better for small datasets)
        epochs=epochs,
        seed=42
    )

    print(f"Model trained with vocabulary size: {len(model.wv)}")
    print(f"Sample words in vocabulary: {list(model.wv.index_to_key[:10])}")

    return model


def solve_analogy(model, word_a, word_b, word_c, topn=5):
    """
    Solve analogy: A - B + C = ?
    Returns the most similar word to the vector: C + (A - B)
    """
    try:
        # Get word vectors
        vec_a = model.wv[word_a]
        vec_b = model.wv[word_b]
        vec_c = model.wv[word_c]

        # Vector arithmetic: D ≈ C + (A - B)
        result_vector = vec_c + (vec_a - vec_b)

        # Find most similar words to result vector
        # Exclude the input words from results
        similar = model.wv.similar_by_vector(result_vector, topn=topn + 3)

        print(f"\nAnalogy: {word_a} - {word_b} + {word_c} = ?")
        print(f"Top {topn} candidates:")

        results = []
        for word, score in similar:
            if word not in [word_a, word_b, word_c]:
                results.append((word, score))
                print(f"  {word}: {score:.4f}")
                if len(results) >= topn:
                    break

        # Return the top result
        if results:
            return results[0][0]
        return None

    except KeyError as e:
        print(f"Error: Word {e} not found in vocabulary!")
        return None


def main():
    # Load corpus
    sentences = load_and_preprocess_corpus('corpus.txt')

    if sentences is None:
        print("Failed to load corpus. Please ensure corpus.txt is in the same directory.")
        return

    # Train Word2Vec model
    model = train_word2vec(sentences, vector_size=100, window=5, min_count=1, epochs=100)

    print("\n" + "=" * 60)
    print("SOLVING ANALOGIES")
    print("=" * 60)

    # Solve analogy 1: doctors - medicine + law = ?
    word1 = solve_analogy(model, 'doctors', 'medicine', 'law', topn=5)

    # Solve analogy 2: teachers - schools + hospitals = ?
    word2 = solve_analogy(model, 'teachers', 'schools', 'hospitals', topn=5)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if word1 and word2:
        flag = f"SIGMOID_{word1.upper()}_{word2.upper()}"
        print(f"\n🚩 FLAG: {flag}")
        return flag
    else:
        print("\nError: Could not solve one or both analogies")
        return None


if __name__ == "__main__":
    flag = main()