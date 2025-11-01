from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# The corpus text
corpus_text = """
Doctors practice medicine daily, relying on medical knowledge to diagnose and treat patients. Medicine forms the foundation of every doctor's expertise and daily practice. Doctors dedicate years to studying medicine before practicing.

Engineers follow law strictly, ensuring compliance with building codes and regulations. Law provides the essential framework that guides all engineering work and decisions. Engineers must master law to practice safely and effectively.

Teachers work in schools, utilizing educational facilities to instruct and guide students. Schools provide the environment where teachers develop and apply their teaching methods. Teachers depend on schools for resources and structure.

Athletes recover in hospitals, receiving specialized treatment for sports injuries and rehabilitation. Hospitals offer the medical expertise athletes need for proper healing and recovery. Athletes rely on hospitals for comprehensive care after intense physical activity.

Medicine enables doctors to provide accurate diagnoses and effective treatments. Law enables engineers to design structures that meet safety standards and regulations.

Schools enable teachers to create engaging learning experiences and educational programs. Hospitals enable athletes to receive expert medical care and rehabilitation services.

Doctors specialize in medicine, focusing on different medical fields and treatments. Engineers specialize in law compliance, understanding different regulatory requirements.

Teachers specialize in school-based education, adapting to various classroom environments. Athletes specialize in hospital recovery programs, following medical protocols.

Medicine guides doctors in their professional practice and ethical decisions. Law guides engineers in their design choices and compliance requirements.

Schools guide teachers in curriculum development and instructional strategies. Hospitals guide athletes in injury recovery and rehabilitation processes.

Doctors apply medicine through patient care and treatment protocols. Engineers apply law through regulatory compliance and safety measures.

Teachers apply educational methods within school settings and systems. Athletes apply recovery techniques under hospital supervision and care.

Medicine serves as doctors' primary tool for healing and patient care. Law serves as engineers' primary framework for professional practice.

Schools serve as teachers' primary workplace and resource center. Hospitals serve as athletes' primary medical care and recovery facility.

Doctors master medicine through extensive study and clinical experience. Engineers master law through regulatory knowledge and compliance expertise.

Teachers master educational techniques within school environments. Athletes master recovery processes through hospital rehabilitation programs.

Medicine defines doctors' professional identity and scope of practice. Law defines engineers' professional responsibilities and limitations.

Schools define teachers' work environment and educational context. Hospitals define athletes' medical care and recovery framework.

Doctors depend on medicine for their professional effectiveness. Engineers depend on law for their professional legitimacy.

Teachers depend on schools for their instructional effectiveness. Athletes depend on hospitals for their physical recovery.

Medicine supports doctors in delivering quality healthcare services. Law supports engineers in maintaining safe construction practices.

Schools support teachers in providing quality education. Hospitals support athletes in achieving full recovery.

Doctors integrate medicine into every aspect of patient care. Engineers integrate law into every aspect of project design.

Teachers integrate educational resources within school systems. Athletes integrate medical care within hospital environments.

Medicine is essential for doctors to practice their profession. Law is essential for engineers to practice their profession.

Schools are essential for teachers to practice their profession. Hospitals are essential for athletes to maintain their health.

Doctors and medicine work together in healthcare delivery. Engineers and law work together in construction projects.

Teachers and schools work together in education. Athletes and hospitals work together in recovery.

Medicine helps doctors heal patients and save lives. Law helps engineers build safely and compliantly.

Schools help teachers educate students effectively. Hospitals help athletes recover from injuries completely.

Doctors use medicine to treat diseases and conditions. Engineers use law to ensure project safety and legality.

Teachers use schools to deliver curriculum and instruction. Athletes use hospitals to receive medical treatment and therapy.

Medicine is the core of doctors' professional training. Law is the core of engineers' professional training.

Schools are the core of teachers' professional environment. Hospitals are the core of athletes' medical support system.

Doctors practice medicine in hospitals and clinics. Engineers practice law in construction and design firms.

Teachers practice education in schools and classrooms. Athletes receive treatment in hospitals and medical centers.

Medicine requires doctors to understand human biology. Law requires engineers to understand regulatory frameworks.

Schools require teachers to understand learning environments. Hospitals require athletes to understand recovery protocols.

Doctors advance medicine through research and practice. Engineers advance law through compliance and innovation.

Teachers advance education through school programs. Athletes advance sports through hospital rehabilitation.

Medicine connects doctors to patient care outcomes. Law connects engineers to project safety outcomes.

Schools connect teachers to student learning outcomes. Hospitals connect athletes to physical recovery outcomes.
"""

# Preprocess the corpus into sentences of tokens
sentences = []
for line in corpus_text.strip().split('\n'):
    if line.strip():
        # Convert to lowercase and tokenize
        tokens = simple_preprocess(line, deacc=True)
        if tokens:
            sentences.append(tokens)

print(f"Number of sentences: {len(sentences)}")
print(f"Sample sentence: {sentences[0][:10]}...")

# Try multiple model configurations to find the best analogies
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
        # positive=[word_a, word_c] means we add vectors for word_a and word_c
        # negative=[word_b] means we subtract vector for word_b
        # This computes: word_a - word_b + word_c
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

# Analogy 1: doctors - medicine + law = ?
print("\nAnalogy 1: doctors - medicine + law = ?")
word1_results = solve_analogy(model, 'doctors', 'medicine', 'law', top_n=15)

# Analogy 2: teachers - schools + hospitals = ?
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