import numpy as np
from PIL import Image

# thanks to
# http://stackoverflow.com/questions/35777830/fast-absolute-difference-of-two-uint8-arrays
def difference_image(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b

def sum_chunk(x, chunk_size, axis=-1):
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.sum(axis=axis+1)

#brut force...
def get_clustered_data_old(image, cluster_size):
    clustered_image = []
    # prvni index je radka (tedy osa y)
    for yr in range(image_size_y / cluster_size):
        clustered_image.append([])
        for xr in range(image_size_x / cluster_size):
            diff_count = 0
            for yc in range(cluster_size):
                for xc in range(cluster_size):
                    diff_count = diff_count + image[yr * cluster_size + yc][xr * cluster_size + xc]
            clustered_image[yr].append(diff_count)
    return clustered_image

# Vraci soucty ctvercu v obrazku pro cluster_size x cluster_size
# normalizovane np.uint16 pokud je cluster_size > 16 (tj. 16x16x256)
def get_clustered_data(image, cluster_size):
    clustered_img = sum_chunk(sum_chunk(image, cluster_size, axis=0), cluster_size, axis=1)
    max_diff = np.amax(clustered_img)
    if max_diff > np.iinfo(np.uint16).max:
        normalized_img = clustered_img.astype(np.float64) / max_diff
        normalized_img = normalized_img * np.iinfo(np.uint16).max
        normalized_img = normalized_img.astype(np.uint16)
        return normalized_img
    else:
        return clustered_img

def read_preprocess_image(filename, cluster_size):
    #image = Image.open(filename)
    #image = np.array(Image.open(filename))
    # !!! v numpy je bug pri konvertu z PIL b/w image do numpy.array
    # (dava to nahodny data, prip. segfaultuje),
    # ale lze to obejit nasledovne
    pil_image = Image.open(filename)
    image = np.reshape(pil_image.getdata(), (pil_image.size[1], pil_image.size[0]))

    #np.savetxt(filename.split("/")[-1:][0] + ".preprocessed0-5", image, fmt='%d')
    if cluster_size > 1:
        image = get_clustered_data(image, cluster_size)

    #np.savetxt(filename.split("/")[-1:][0] + ".preprocessed1-5", image, fmt='%d')
    return image

def get_image_label(filename):
    label = 1 #positive alarm
    # podle prviho pismena posledni slozky souboru pozname true/false
    if filename.split("/")[-2:][0][0] == 'f': #false
        label = 0 #negative alarm
    return label

