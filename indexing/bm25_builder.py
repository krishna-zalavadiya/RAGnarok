import collections
import re
import math
import json

# ---------------------------------------------------------
# 1. DYNAMIC JSON GRAPH FLATTENING ENGINE
# ---------------------------------------------------------
def flatten_json_recursive(data, weight_map=None, current_path=""):
    """
    Recursively traverses and flattens every single string element within the 
    entire JSON structure. No keys are ignored or selectively skipped.
    """
    if weight_map is None:
        weight_map = collections.defaultdict(float)
        
    if isinstance(data, dict):
        for key, value in data.items():
            # Process string keys as query token hooks
            normalized_key = key.lower().strip()
            weight_map[normalized_key] += 1.0
            
            # Recursive dive into nested structures
            flatten_json_recursive(value, weight_map, current_path=f"{current_path}.{key}")
            
    elif isinstance(data, list):
        for item in data:
            flatten_json_recursive(item, weight_map, current_path)
            
    elif isinstance(data, (str, int, float)):
        # Normalize and track raw text property tokens safely
        normalized_val = str(data).lower().strip()
        weight_map[normalized_val] += 0.5  # Universal expansion token weight
        
    return weight_map

def load_and_map_full_ontology(file_path="skill_map.json"):
    """Reads the entire schema without selective filtering blocks."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)
    
    # Flatten everything into a single operational mapping matrix
    return flatten_json_recursive(raw_json_data)

# Extract and instantiate the entire content matrix dynamically
try:
    FULL_ONTOLOGY_MATRIX = load_and_map_full_ontology("skill_map.json")
except FileNotFoundError:
    print("[WARNING] 'skill_map.json' missing. Using an empty matrix layer.")
    FULL_ONTOLOGY_MATRIX = {}

# ---------------------------------------------------------
# 2. COMPLETE OKAPI BM25 INVERTED INDEX ENGINE
# ---------------------------------------------------------
class FullTextBM25Index:
    """Okapi BM25 implementation handling token arrays seamlessly."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_lengths = [len(doc) for doc in corpus]
        self.doc_freqs = collections.Counter()
        self.inverted_index = collections.defaultdict(dict)
        self._build_index(corpus)

    def _build_index(self, corpus):
        for idx, doc in enumerate(corpus):
            counts = collections.Counter(doc)
            for token, freq in counts.items():
                self.inverted_index[token][idx] = freq
                self.doc_freqs[token] += 1

    def get_scores(self, query_tokens):
        scores = collections.defaultdict(float)
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            
            for doc_idx, freq in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf * (numerator / denominator)
        return scores

# ---------------------------------------------------------
# 3. TEXT HANDLING UTILITIES
# ---------------------------------------------------------
def normalize_text(text):
    """Tokenization engine parsing alphanumeric & special programming text hooks."""
    if not text:
        return []
    text = text.lower()
    return re.findall(r'\b[a-z0-9\+\#\-\.]+\b', text)

def get_fully_expanded_query(user_query):
    """
    Expands the query token stream using the entire structural schema matrix 
    extracted from the full file.
    """
    tokens = normalize_text(user_query)
    expanded_tokens = collections.defaultdict(float)
    
    for token in tokens:
        # Step A: Give the base keyword maximum matching value
        expanded_tokens[token] += 1.0
        
        # Step B: Loop over the full matrix map to capture structural relationships
        for entity, weight in FULL_ONTOLOGY_MATRIX.items():
            if token in entity or entity in token:
                # Capture structural synonyms/links anywhere in the JSON data stream
                expanded_tokens[entity] += (weight * 0.5)
                
    return expanded_tokens

# ---------------------------------------------------------
# 4. COMPREHENSIVE MATCH PIPELINE ENGINE
# ---------------------------------------------------------
def rank_candidates_with_full_matrix(job_description, candidates):
    """Ranks candidates across the full text index with no selective parameters."""
    # 1. Expand query utilizing all tokens extracted from the file structure
    query_expansion = get_fully_expanded_query(job_description)
    query_tokens = list(query_expansion.keys())
    
    # 2. Clean candidate profile logs
    tokenized_resumes = [normalize_text(c["resume_text"]) for c in candidates]
    
    # 3. Process the Inverted BM25 Index matching pass
    engine = FullTextBM25Index(tokenized_resumes)
    lexical_scores = engine.get_scores(query_tokens)
    
    ranked_pipeline = []
    
    # 4. Structural Verification & Alignment Calculation Pass
    for idx, cand in enumerate(candidates):
        base_score = lexical_scores.get(idx, 0.0)
        cand_tokens_set = set(tokenized_resumes[idx])
        
        # Award dynamic structural score points if other related keywords overlap
        co_occurrence_credit = 0.0
        for token in query_tokens:
            if token in cand_tokens_set:
                co_occurrence_credit += 0.10
                
        final_score = base_score + co_occurrence_credit
        ranked_pipeline.append({
            "id": cand["id"],
            "name": cand["name"],
            "final_score": round(final_score, 4),
            "lexical_score": round(base_score, 4)
        })
        
    return sorted(ranked_pipeline, key=lambda x: x["final_score"], reverse=True)