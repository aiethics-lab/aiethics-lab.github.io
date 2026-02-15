"""
Generate a 5,000-word GloVe 50d subset JSON file for client-side use.
Downloads GloVe 6B from Stanford NLP if not already present.
"""

import json
import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_TXT = "glove.6B.50d.txt"
OUTPUT_FILE = "glove-50d-5k.json"
VOCAB_SIZE = 5000

# Words we absolutely need for the ethics toolkit demos
REQUIRED_WORDS = {
    # Vector arithmetic demos
    'king', 'queen', 'man', 'woman', 'prince', 'princess',
    'boy', 'girl', 'father', 'mother', 'son', 'daughter',
    'brother', 'sister', 'husband', 'wife', 'uncle', 'aunt',
    'grandfather', 'grandmother', 'lord', 'lady', 'duke', 'duchess',
    'emperor', 'empress', 'sir', 'madam',
    # Gender terms
    'he', 'she', 'him', 'her', 'his', 'hers', 'male', 'female',
    # Countries & capitals
    'france', 'paris', 'germany', 'berlin', 'japan', 'tokyo',
    'italy', 'rome', 'spain', 'madrid', 'china', 'beijing',
    'russia', 'moscow', 'brazil', 'india', 'delhi', 'canada', 'ottawa',
    # Professions (for bias detection)
    'doctor', 'nurse', 'engineer', 'teacher', 'programmer', 'scientist',
    'lawyer', 'pilot', 'mechanic', 'secretary', 'professor', 'surgeon',
    'accountant', 'architect', 'carpenter', 'chef', 'dentist',
    'electrician', 'firefighter', 'journalist', 'librarian', 'manager',
    'musician', 'painter', 'pharmacist', 'plumber', 'police',
    'receptionist', 'soldier', 'veterinarian',
    # Animals
    'cat', 'dog', 'horse', 'bird', 'fish', 'lion', 'tiger', 'bear',
    'wolf', 'eagle', 'snake', 'rabbit',
    # Emotions
    'happy', 'sad', 'angry', 'afraid', 'surprised', 'calm', 'excited',
    'anxious', 'proud', 'love', 'hate', 'fear', 'joy',
    # Food
    'pizza', 'sushi', 'pasta', 'rice', 'bread', 'cheese',
    # Verb tenses
    'walking', 'walked', 'running', 'ran', 'swimming', 'swam',
    'flying', 'flew',
    # Additional useful words
    'computer', 'technology', 'algorithm', 'data', 'intelligence',
    'artificial', 'ethics', 'bias', 'fair', 'unfair', 'justice',
    'equality', 'freedom', 'privacy', 'safety', 'risk',
}


def download_glove():
    """Download GloVe 6B zip file."""
    if os.path.exists(GLOVE_TXT):
        print(f"✓ {GLOVE_TXT} already exists, skipping download.")
        return

    if not os.path.exists(GLOVE_ZIP):
        print(f"Downloading {GLOVE_URL} (~862 MB)...")
        print("This may take a few minutes depending on your connection.")

        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP, reporthook)
        print("\n✓ Download complete.")

    print(f"Extracting {GLOVE_TXT} from zip...")
    with zipfile.ZipFile(GLOVE_ZIP, 'r') as zf:
        zf.extract(GLOVE_TXT, '.')
    print(f"✓ Extracted {GLOVE_TXT}")


def generate_subset():
    """Extract top VOCAB_SIZE words + required words from GloVe."""
    print(f"Reading {GLOVE_TXT}...")
    embeddings = {}
    line_count = 0

    with open(GLOVE_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            # First VOCAB_SIZE words by frequency (GloVe file is sorted by frequency)
            # Plus all required words
            if line_count < VOCAB_SIZE or word in REQUIRED_WORDS:
                vec = [round(float(x), 6) for x in parts[1:]]
                if len(vec) == 50:
                    embeddings[word] = vec
            line_count += 1

            if line_count % 50000 == 0:
                print(f"  Processed {line_count:,} lines, {len(embeddings):,} words kept...")

    print(f"✓ Total words in subset: {len(embeddings):,}")

    # Verify required words
    missing = REQUIRED_WORDS - set(embeddings.keys())
    if missing:
        print(f"⚠ Missing required words: {missing}")
    else:
        print("✓ All required words found in GloVe vocabulary.")

    # Write JSON
    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(embeddings, f, separators=(',', ':'))

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✓ Wrote {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    download_glove()
    generate_subset()
    print("\nDone! The JSON file is ready for use in the Word Embeddings tool.")
