import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''
def showallframes(vid, adv_vid, model_name):
    idx=[1, 3, 5, 7, 9, 11, 13, 15]
    plt.figure()
    for i in range(0,8):
        indx=idx[i]
        image = vid[indx]
        if model_name=='c3d':
            image = np.transpose(image,(1,2,0))  # put channel last
        else:
            image=np.transpose(image,(1,2,0))
        plt.subplot(2,8,(i+1))
        plt.imshow(image)
        plt.axis('off')
        plt.title('clean')
    for i in range(0,8):
        indx=idx[i]
        image = adv_vid[indx]
        if model_name=='c3d':
            image = np.transpose(image,(1,2,0))  # put channel last
        else:
            image=np.transpose(image,(1,2,0))
        plt.subplot(2,8,(i+9))
        plt.imshow(image)
        plt.axis('off')
        plt.title('adv')
    plt.show()
'''


def showallframes(vid, adv_vid, model_name):
    idx=[1, 3, 5, 7, 9, 11, 13, 15]
    plt.figure()
    for i in range(0,8):
        indx=idx[i]
        image = adv_vid[indx]
        if model_name=='c3d':
            image = np.transpose(image,(1,2,0))  # put channel last
        else:
            image=np.transpose(image,(1,2,0))
        plt.subplot(1,8,i+1)
        img = Image.fromarray(np.uint8(image*255))
        # specify image quality img.save('new1.jpg',quality=95)，default as 75
        img.save('C:/Users/FenHua/Desktop/EARL/Utils/results/{}.png'.format(i+1),quality=99)
        plt.imshow(image)
        plt.axis('off')
        #plt.title('adv')
    plt.show()