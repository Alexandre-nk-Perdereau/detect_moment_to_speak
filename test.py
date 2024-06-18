import numpy as np

# Charger les fichiers segments et labels
segments = np.load('output/segments.npy', allow_pickle=True)
labels = np.load('output/segment_labels.npy', allow_pickle=True)

print(f"Segments shape: {segments.shape}")
print(f"Labels shape: {labels.shape}")

# Vérification de la correspondance des dimensions
if len(segments) != len(labels):
    print("Erreur : Le nombre de segments et de labels ne correspond pas.")
else:
    print("Les dimensions des segments et des labels correspondent.")
