import os
import kagglehub

# Download the dataset to the local dataset/ directory
dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(dataset_dir, exist_ok=True)

path = kagglehub.dataset_download(
    "alfrandom/protein-secondary-structure",
    output_dir=dataset_dir,
    force_download=True,
)

print("Dataset saved to:", path)
