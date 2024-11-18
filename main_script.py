import pickle
atlas_path = 'ViT-V-Net/IXI_data/atlas.pkl'
with open(atlas_path, 'rb') as f:
    atlas_data = pickle.load(f)
    print(type(atlas_data))
    print(atlas_data)