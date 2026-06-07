import math

def build_vocabulary(docs):
    """
    Takes a list of documents (strings of space-separated tags) and 
    returns a sorted list of all unique tags across all documents.
    """
    unique_tags = set()  # A set automatically handles duplicates
    
    for doc in docs:
        # Split each document string into individual words
        words = doc.split()
        # Add the words to our set of unique tags
        unique_tags.update(words)
        
    # Convert the set back to a list and sort it alphabetically
    return sorted(list(unique_tags))
def compute_tf(doc):
    """
    Computes Term Frequency (TF) for a single document.

    TF(tag) = (Number of times tag appears in the document)
              / (Total number of tags in the document)

    Args:
        doc: A string of space-separated tags (e.g. "python ml python data python").

    Returns:
        A dictionary where each key is a unique tag and each value is its TF score.
        All values sum to 1.0.
    """
    tags = doc.split()
    total_tags = len(tags)
    return {tag: tags.count(tag) / total_tags for tag in set(tags)}    
def compute_idf(docs):
    # STEP 1: Count how many docs contain each tag
    doc_count = {}
    for doc in docs:
        unique_tags = set(doc.split())
        for tag in unique_tags:
            doc_count[tag] = doc_count.get(tag, 0) + 1  # just counting here!

    # STEP 2: Apply the formula AFTER counting is done
    total_docs = len(docs)
    idf = {}
    for tag in doc_count:
        idf[tag] = math.log(total_docs / doc_count[tag])
    
    return idf
def compute_tfidf(doc, docs):
    # Step 1: Get TF scores for this single document
    tf = compute_tf(doc)
    
    # Step 2: Get IDF scores for all tags across all documents
    idf = compute_idf(docs)
    
    # Step 3: Multiply them together — but ONLY for tags in this document
    tfidf = {tag: tf[tag] * idf[tag] for tag in tf}

    return tfidf

def compute_cosine_similarity(vec_a, vec_b):
    """
    Computes the cosine similarity between two vectors.

    Cosine Similarity = (A · B) / (||A|| × ||B||)

    Args:
        vec_a: A list of numbers (first vector).
        vec_b: A list of numbers (second vector).

    Returns:
        A float between 0 and 1 representing how similar the two vectors are.
        Returns 0.0 if either vector is all zeros.
    """
    # Step 1: Dot product — multiply matching slots, then sum
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Step 2: Magnitude of each vector — square root of sum of squares
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # Edge case: if either vector is all zeros, return 0.0
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Step 3: Final cosine similarity
    return dot_product / (magnitude_a * magnitude_b)


# --- Example Usage ---
docs = [
    "python ml statistics data",
    "cloud aws networking infrastructure",
    "python docker cloud devops"
]

vocabulary = build_vocabulary(docs)
doc = "python ml python data python"
tf = compute_tf(doc)
idf = compute_idf(docs)
tfidf = compute_tfidf(doc, docs)
print("Vocabulary:", vocabulary)
print("TF:", tf)
print("IDF:", idf)
print("TF-IDF:", tfidf)

# --- Task 5 Test ---
vec_a = [0, 0, 0.2, 0, 0.3, 0.5]
vec_b = [0, 0, 0.1, 0, 0.4, 0.5]
print("Cosine Similarity:", round(compute_cosine_similarity(vec_a, vec_b), 3))