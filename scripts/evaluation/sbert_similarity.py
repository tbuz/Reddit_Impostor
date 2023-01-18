from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS

model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['The quick fox jumps over the lazy cat',
    'The quick fox jumps over the lazy cat',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

similarity_array = cosine_similarity([embeddings[0]], embeddings[1:])
print(similarity_array)

n_components = 2
tsne = TSNE(n_components=n_components)
tsne_result = tsne.fit_transform(embeddings)
print(tsne_result.shape)

optics_clustering = OPTICS(min_samples=3).fit(tsne_result)
print(optics_clustering.labels_)

    