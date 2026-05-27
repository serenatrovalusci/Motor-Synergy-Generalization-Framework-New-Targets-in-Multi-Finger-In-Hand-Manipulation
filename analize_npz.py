import numpy as np

data = np.load("synergies/recon_5_pca.npz", allow_pickle=True)

print("Keys inside the npz file:")
print(data.files)

for key in data.files:
     arr = data[key]
     print(f"\nKey: {key}", arr)

print(data['activities'].shape)
print(data['recon'].shape)
print(data['original'].shape)
print(data['n_synergies'].shape[0])

    
#     if isinstance(arr, np.ndarray):
#         print(f"  Shape: {arr.shape}")
#         print(f"  Dtype: {arr.dtype}")


# arr = data['actions']

# print(f"\n {key} contains object data")
# print("First element:", arr)

