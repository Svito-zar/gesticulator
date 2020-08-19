from joblib import dump, load
import numpy as np
from sklearn.decomposition import PCA

root_dir = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/Play"

train_poses = np.load(root_dir + '/Y_train.npy').astype(np.float32)

print(train_poses.shape)

flat_pose = train_poses.reshape(-1, train_poses.shape[2]) # flatten sequences

print(flat_pose.shape)

"""pca = PCA(n_components=12)
pca.fit(flat_pose)

print(np.cumsum(pca.explained_variance_ratio_))

dump(pca, 'pca_model_12.joblib')"""


pca = load('pca_model_12.joblib')

print(flat_pose[0])

train_pca = pca.transform(flat_pose)

final_train = train_pca.reshape(train_poses.shape[0], train_poses.shape[1], -1)

print(final_train.shape)

np.save("Y_train.npy", final_train)

### DEV ###

dev_poses = np.load(root_dir + '/Y_dev.npy').astype(np.float32)

flat_dev = dev_poses.reshape(-1, train_poses.shape[2]) # flatten sequences


dev_pca = pca.transform(flat_dev)

final_dev = dev_pca.reshape(dev_poses.shape[0], dev_poses.shape[1], -1)

print(dev_poses.shape)
print(final_dev.shape)

np.save("Y_dev.npy", final_dev)


