__author__ = 'ahmaddorri'
import numpy as np
import pickle

pickle_file = 'gray.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_label = save['train_label']
    test_dataset = save['test_dataset']
    test_label = save['test_label']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_label.shape)
    print('Test set', test_dataset.shape, test_label.shape)



def GCN(image, min_divisor=1e-4):
    """Global Contrast Normalization"""

    imsize = image.shape[0]
    mean = np.mean(image, axis=(1,2), dtype=float)
    std = np.std(image, axis=(1,2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.
    image_GCN = np.zeros(image.shape, dtype=float)

    for i in np.arange(imsize):
        image_GCN[i,:,:] = (image[i,:,:] - mean[i]) / std[i]

    return image_GCN

train_data_GCN = GCN(train_dataset[0:1000])
test_data_GCN = GCN(test_dataset[0:1000])

for i in range(1,73):
    train_data_GCN=np.append(train_data_GCN,GCN(train_dataset[1000*i:1000*(i+1)]),axis=0)

for i in range(1,26):
    test_data_GCN=np.append(test_data_GCN,GCN(test_dataset[1000*i:1000*(i+1)]),axis=0)

train_data_GCN=np.append(train_data_GCN,GCN(train_dataset[73000:]),axis=0)
test_data_GCN=np.append(test_data_GCN,GCN(test_dataset[26000:]),axis=0)

print(train_data_GCN.shape, train_label.shape)
print(test_data_GCN.shape, test_label.shape)

pickle_file = 'GCN.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_data_GCN,
    'train_label': train_label,
    'test_dataset': test_data_GCN,
    'test_label': test_label,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)

