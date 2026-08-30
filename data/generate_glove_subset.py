"""
Build the GloVe subset used by the Word Embeddings Workbench.

Output is a binary Float32 matrix plus a JSON vocabulary, not one big JSON
object. The previous version emitted 5,000 words of 50d vectors as JSON, which
cost 2.1 MB for a vocabulary small enough that most of the WEAT female target
names fell outside it - the association test was silently running against a
single word. Numbers written as decimal text also cost roughly nine bytes each
to store four bytes of information, and JSON.parse of a multi-megabyte string
blocks the main thread.

Two tiers are produced so the workbench is usable immediately and so the cost
of a larger model is something a student can measure rather than be told:

    glove-small.bin    5,000 words x 50d    ~1 MB
    glove-large.bin    20,000 words x 100d  ~7.6 MB

The binary layout for each is vocab_size * dim float32, little-endian and
row-major, paired with a JSON vocabulary in row order. Loading is a fetch plus
a Float32Array view, with no parsing step.

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

# Two tiers, so the workbench is usable the moment the page opens and the cost
# of a bigger model becomes something a student can feel rather than be told.
# The small tier loads in well under a second; the large one is fetched on
# demand, and the two can then be compared directly in the tool.
TIERS = [
    # The small tier pins only CORE_WORDS. It is otherwise a straight frequency
    # cut, so it has the coverage gaps a 5,000-word vocabulary really has -
    # including most of the female names in Caliskan's WEAT 6. That is the
    # point: the tool reports the test as incomplete instead of quietly running
    # it on whatever words happen to be present, which is what the old
    # single-tier build did without saying so.
    {"dim": 50,  "vocab": 5000,  "suffix": "small", "extended": False},
    {"dim": 100, "vocab": 20000, "suffix": "large", "extended": True},
]

# Pinned into EVERY tier: without these the basic demos cannot run at all.
CORE_WORDS = {
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

# Pinned into the LARGE tier only. These are the word sets the published
# methods require; leaving them out of the small tier is deliberate, so the
# tool can show what a limited vocabulary costs you rather than assert it.
EXTENDED_WORDS = {
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
}


def download_glove(dims):
    needed = [f"glove.6B.{d}d.txt" for d in dims]
    if all(os.path.exists(n) for n in needed):
        print("  source files already extracted.")
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
    with zipfile.ZipFile(GLOVE_ZIP) as zf:
        for name in needed:
            if not os.path.exists(name):
                print(f"Extracting {name}...")
                zf.extract(name, ".")


def build(dim, vocab_size, suffix, extended):
    required = CORE_WORDS | EXTENDED_WORDS if extended else CORE_WORDS
    src = f"glove.6B.{dim}d.txt"
    out_bin = f"glove-{suffix}.bin"
    out_vocab = f"glove-{suffix}.json"

    print(f"\nBuilding '{suffix}' tier: {vocab_size:,} words x {dim}d from {src}")
    words, vectors = [], []
    seen = set()
    kept_required = set()

    with open(src, "r", encoding="utf-8") as fh:
        for rank, line in enumerate(fh):
            parts = line.rstrip().split(" ")
            word = parts[0]
            if len(parts) != dim + 1:
                continue
            is_required = word in required
            if rank >= vocab_size and not is_required:
                continue
            if word in seen:
                continue
            seen.add(word)
            if is_required:
                kept_required.add(word)
            words.append(word)
            vectors.append([float(x) for x in parts[1:]])

    missing = required - kept_required
    if missing:
        print(f"  WARNING: {len(missing)} required words absent: {sorted(missing)[:10]}")

    with open(out_bin, "wb") as fh:
        for vec in vectors:
            fh.write(struct.pack(f"<{dim}f", *vec))

    with open(out_vocab, "w") as fh:
        json.dump({"dim": dim, "count": len(words), "tier": suffix,
                   "words": words}, fh, separators=(",", ":"))

    mb = os.path.getsize(out_bin) / 1048576
    print(f"  {len(words):,} words -> {out_bin} ({mb:.1f} MB) + {out_vocab}")
    return {"tier": suffix, "dim": dim, "count": len(words), "mb": round(mb, 2),
            "extendedVocabulary": extended}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_glove([t["dim"] for t in TIERS])
    manifest = [build(t["dim"], t["vocab"], t["suffix"], t["extended"]) for t in TIERS]
    with open("glove-tiers.json", "w") as fh:
        json.dump({"tiers": manifest}, fh, indent=2)
    print("\nwrote glove-tiers.json")
    print("Delete glove.6B.zip and the extracted .txt files when finished;")
    print("they are build inputs and must not be committed.")
