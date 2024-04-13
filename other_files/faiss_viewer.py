import faiss
import numpy as np

def load_faiss_index(file_path):
    # Load the FAISS index from the file
    index = faiss.read_index(file_path)
    return index

def inspect_index(index):
    print("Number of vectors:", index.ntotal)
    print("Dimension of the vectors (if homogeneous):", index.d)

def print_all_vectors(index):
    if isinstance(index, faiss.IndexFlat):
        print("Vectors stored in the index:")
        for i in range(index.ntotal):
            vector = np.zeros((index.d,), dtype='float32')
            index.reconstruct(i, vector)
            print(f"Vector {i}: {vector}")
    else:
        print("This type of index does not support direct vector extraction or reconstruction.")

def search_in_index(index, vector, k=5):
    """ Search the index for the top k nearest vectors to the provided vector """
    # Ensure the vector is in the form of a numpy array and has the right dimensions
    vector = np.array(vector, dtype='float32').reshape(1, -1)
    # Search the index
    distances, indices = index.search(vector, k)
    return distances, indices

# Example usage
index_path = '../output/FAISS/sentence-transformers/all-mpnet-base-v2/index.faiss'  # Adjust this path to where your .faiss file is saved
index = load_faiss_index(index_path)

# Example vector (must be the same dimension as those in the index)
query_vector = np.random.random(index.d).astype('float32')
distances, indices = search_in_index(index, query_vector, k=5)
inspect_index(index)

print("Distances:", distances)
print("Indices:", indices)

# inspect_index(index)
# print_all_vectors(index)