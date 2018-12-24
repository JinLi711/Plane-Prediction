from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def view_1_image (json, index, save=False):
    """
    View 1 plane image and save it. 
    Also print out its features (label, coordinate, id).

    :param json: json items
    :type  json: dict
    :param index: The index of a json file for the plane image
    :type  index: int
    :param save: Save it or not
    :type  save: bool
    """

    im = np.array(json['data'][index]).astype('uint8')
    im = im.reshape((3, 400)).T.reshape((20,20,3))
    plt.imshow(im)
    plt.show()
    #print(im.shape)
    
    label = json['labels'][index] 
    loc = json['locations'][index]
    scene_id = json['scene_ids'][index]
    
    if label == 1:
        print ('Is Plane: YES')
    else:
        print ('Is Plane: NO')
        
    print ('Longitude: ' + str (loc[0]))
    print ('Latitude: ' + str (loc[1]))
    print ('ID: ' + scene_id)
    
    if save:
        out_im = Image.fromarray(im)
        out_im.save('images/image{}.png'.format (str (label) + "_" + str (index)))


def plot_multiple_images (json, cols, rows, label):
    """
    Show multiple images of the data set.
    
    :param json: Image items
    :type  json: dict
    :param cols: Number of columns of images to be shown
    :type  cols: int 
    :param rows: Number of rows of images to be shown
    :type  rows: int
    :param label: What type of images to be shown 
        (planes, geography, partial planes, confusion items)
    :type  label: str
    """
    
    import random
    
    planesR = [0,8000] # range of plane images
    geoR = [8000, 16000] # range of geography images
    pplanesR = [16000, 24000] # range of partial planes
    confR = [24000, 32000] # range of images that look like planes
    
    if label == 'planes':
        indexes = planesR
    elif label == 'geography':
        indexes = geoR
    elif label == 'partial planes':
        indexes = pplanesR
    elif label == 'confusion items':
        indexes = confR
    else:
        raise Exception ('Not a category')
        
    fig=plt.figure(figsize=(10, 10))
    for i in range(1, rows * cols +1):
        randn = random.randrange(indexes[0],indexes[1])
        img = np.array(json['data'][randn]).astype('uint8')
        img = img.reshape((3, 400)).T.reshape((20,20,3))
        fig.add_subplot(rows, cols, i)
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
    plt.show() 
