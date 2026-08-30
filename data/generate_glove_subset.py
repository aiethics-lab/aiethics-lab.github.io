"""
Build the GloVe subset used by the Word Embeddings Workbench.

Output is a binary Float32 matrix plus a JSON vocabulary, not one big JSON
object. The previous version emitted 5,000 words of 50d vectors as JSON, which
cost 2.1 MB for a vocabulary small enough that most of the WEAT female target
names fell outside it - the association test was silently running against a
single word. Numbers written as decimal text also cost roughly nine bytes each
to store four bytes of information, and JSON.parse of a multi-megabyte string
blocks the main thread.

The binary layout is:

    glove-100d.bin    vocab_size * 100 float32, little-endian, row-major
    glove-100d.json   { dim, count, words: [...] }  vocabulary in row order

At 20,000 words that is 8 MB of vectors, fetched once and then cached. Loading
is a fetch plus a Float32Array view, with no parsing step.

Downloads GloVe 6B from Hugging Face if the source file is not already present.
The 862 MB archive and the extracted text file are build inputs, not artefacts:
delete them afterwards and keep them out of git.

Usage:  python3 generate_glove_subset.py
"""

import json
import os
import struct
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
DIM = 100
GLOVE_TXT = f"glove.6B.{DIM}d.txt"
OUT_BIN = f"glove-{DIM}d.bin"
OUT_VOCAB = f"glove-{DIM}d.json"

# Words beyond the frequency cut that the toolkit needs regardless. Anything
# listed here is kept even if it is rarer than the cut-off, because a demo that
# silently drops half its word set produces a degenerate result rather than an
# error.
VOCAB_SIZE = 20000

REQUIRED_WORDS = {
    # Vector arithmetic demos
    'king', 'queen', 'man', 'woman', 'prince', 'princess',
    'boy', 'girl', 'father', 'mother', 'son', 'daughter',
    'brother', 'sister', 'husband', 'wife', 'uncle', 'aunt',
    'grandfather', 'grandmother', 'lord', 'lady', 'duke', 'duchess',
    'emperor', 'empress', 'sir', 'madam', 'himself', 'herself',
    # Gender terms
    'he', 'she', 'him', 'her', 'his', 'hers', 'male', 'female',
    # Countries and capitals
    'france', 'paris', 'germany', 'berlin', 'japan', 'tokyo',
    'italy', 'rome', 'spain', 'madrid', 'china', 'beijing',
    'russia', 'moscow', 'brazil', 'brasilia', 'india', 'delhi',
    'canada', 'ottawa', 'egypt', 'cairo', 'kenya', 'nairobi',
    # Professions, for the bias chart
    'doctor', 'nurse', 'engineer', 'teacher', 'programmer', 'scientist',
    'lawyer', 'pilot', 'mechanic', 'secretary', 'professor', 'surgeon',
    'accountant', 'architect', 'carpenter', 'chef', 'dentist',
    'electrician', 'firefighter', 'journalist', 'librarian', 'manager',
    'musician', 'painter', 'pharmacist', 'plumber', 'police',
    'receptionist', 'soldier', 'veterinarian', 'nanny', 'midwife',
    'hairdresser', 'janitor', 'therapist', 'economist', 'physicist',
    # WEAT target and attribute sets (Caliskan et al. 2017, tests 6-8)
    'amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna',
    'mary', 'elizabeth', 'maria', 'susan', 'barbara', 'sharon', 'nancy',
    'karen', 'betty', 'helen', 'rebecca', 'julia', 'emily', 'laura',
    'john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill',
    'brian', 'ronald', 'david', 'james', 'robert', 'michael', 'william',
    'executive', 'management', 'corporation', 'salary', 'office',
    'business', 'career', 'professional',
    'parents', 'children', 'cousins', 'marriage', 'wedding', 'relatives',
    'household', 'kids', 'home', 'family',
    'math', 'algebra', 'geometry', 'calculus', 'equations', 'computation',
    'numbers', 'addition', 'poetry', 'art', 'dance', 'literature', 'novel',
    'symphony', 'drama', 'sculpture',
    'science', 'physics', 'chemistry', 'einstein', 'nasa', 'experiment',
    'astronomy',
    # Animals, emotions, food, tenses for the explorer tabs
    'cat', 'dog', 'horse', 'bird', 'fish', 'lion', 'tiger', 'bear',
    'wolf', 'eagle', 'snake', 'rabbit',
    'happy', 'sad', 'angry', 'afraid', 'surprised', 'calm', 'excited',
    'anxious', 'proud', 'love', 'hate', 'fear', 'joy',
    'pizza', 'sushi', 'pasta', 'rice', 'bread', 'cheese',
    'walking', 'walked', 'running', 'ran', 'swimming', 'swam',
    'flying', 'flew',
    # Ethics vocabulary
    'computer', 'technology', 'algorithm', 'data', 'intelligence',
    'artificial', 'ethics', 'bias', 'fair', 'unfair', 'justice',
    'equality', 'freedom', 'privacy', 'safety', 'risk', 'harm',
    'consent', 'accountability', 'transparency', 'discrimination',
}


def download_glove():
    if os.path.exists(GLOVE_TXT):
        print(f"  {GLOVE_TXT} already extracted.")
        return
    if not os.path.exists(GLOVE_ZIP):
        print(f"Downloading {GLOVE_URL} (~862 MB)...")

        def report(count, block, total):
            done = count * block
            pct = min(100, int(done * 100 / total)) if total > 0 else 0
            sys.stdout.write(f"\r  {pct}%  ({done / 1048576:.0f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP, report)
        print()
    print(f"Extracting {GLOVE_TXT}...")
    with zipfile.ZipFile(GLOVE_ZIP) as zf:
        zf.extract(GLOVE_TXT, ".")


def build():
    print(f"Reading {GLOVE_TXT}...")
    words, vectors = [], []
    seen = set()
    kept_required = set()

    with open(GLOVE_TXT, "r", encoding="utf-8") as fh:
        for rank, line in enumerate(fh):
            parts = line.rstrip().split(" ")
            word = parts[0]
            if len(parts) != DIM + 1:
                continue
            required = word in REQUIRED_WORDS
            if rank >= VOCAB_SIZE and not required:
                continue
            if word in seen:
                continue
            seen.add(word)
            if required:
                kept_required.add(word)
            words.append(word)
            vectors.append([float(x) for x in parts[1:]])

    print(f"  kept {len(words):,} words at {DIM}d")

    missing = REQUIRED_WORDS - kept_required
    if missing:
        print(f"  WARNING: {len(missing)} required words absent from GloVe: {sorted(missing)}")
    else:
        print("  all required words present")

    with open(OUT_BIN, "wb") as fh:
        for vec in vectors:
            fh.write(struct.pack(f"<{DIM}f", *vec))

    with open(OUT_VOCAB, "w") as fh:
        json.dump({"dim": DIM, "count": len(words), "words": words}, fh, separators=(",", ":"))

    bin_mb = os.path.getsize(OUT_BIN) / 1048576
    vocab_mb = os.path.getsize(OUT_VOCAB) / 1048576
    print(f"  wrote {OUT_BIN} ({bin_mb:.1f} MB) and {OUT_VOCAB} ({vocab_mb:.2f} MB)")
    print("\nDelete glove.6B.zip and the extracted .txt when finished; they are")
    print("build inputs and must not be committed.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_glove()
    build()
