#Since I ran this on google collab
!pip install youtube-transcript-api sentence-transformers faiss-cpu

from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
 
VIDEO_ID = "oNI-n1b8lxM" #get the video ID from the youtube URL

try:
    transcript = YouTubeTranscriptApi.get_transcript(VIDEO_ID)
except Exception as e:
    print(f"Transcript error: {e}")
    transcript = []

if not transcript:
    print("No transcript found. Exiting.")
    exit()

transcript_chunks = []
for i in range(0, len(transcript), 3):
    segment = transcript[i:i+3]
    text = " ".join([s['text'] for s in segment]).strip()
    if not text:
        continue 
    start = segment[0]['start']
    transcript_chunks.append({'text': text, 'start': start})

if not transcript_chunks:
    print("No text transcript_chunks generated.")
    exit()

print("Loading local model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
texts = [transcript_chunk['text'] for transcript_chunk in transcript_chunks]
embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype('float32')
if embeddings.ndim != 2:
    print("Embeddings are not 2D. Exiting.")
    exit()

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    D, I = index.search(query_embedding, top_k)
    return [transcript_chunks[i] for i in I[0]]

query = input("Enter your search query: ")
results = search(query)

print("\nTop Results:")
for result in results:
    timestamp = int(result['start'])
    print(f"\n Time: {timestamp}s")
    print(f"Text: {result['text']}")
    print(f"Link: https://www.youtube.com/watch?v={VIDEO_ID}&t={timestamp}s")
