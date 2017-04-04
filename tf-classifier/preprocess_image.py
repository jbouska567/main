import numpy as np
from PIL import Image


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

def get_clustered_data(image, cluster_size):
    return sum_chunk(sum_chunk(image, cluster_size, axis=0), cluster_size, axis=1)

def read_preprocess_image(filename, cluster_size):
    #image = Image.open(filename)
    #image = np.array(Image.open(filename))
    # !!! v numpy je bug pri konvertu z PIL b/w image do numpy.array
    # ale lze to obejit nasledovne (dava to nahodny data, prip. sefaultuje)
    pil_image = Image.open(filename)
    #image = np.reshape(pil_image.getdata(), (pil_image.size[1], pil_image.size[0]))
    # TODO ma smysl to konvertovat na 0/1 (z 0/255)?
    image = np.reshape(pil_image.getdata(), (pil_image.size[1], pil_image.size[0])) / 255
    #image = image.astype(bool)
    #print image.shape

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

