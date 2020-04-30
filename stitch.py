import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import os
import argparse

class Stitch():

    def __init__(self, images, detection, focal_length, rectangle):
        self.images = images
        self.shape = images[0].shape[0:2]
        self.detection = detection
        self.featrue = None
        self.focal_length = focal_length
        self.rectangle = rectangle
        self.panorama = None

    def feature_detect(self, images):
        pass

    def feature_matching(self, feature):
        pass

    def cylindrical(self):
        index_array = np.indices(self.shape)
        x = self.focal_length * np.arctan(index_array[0]/self.focal_length)
        y = np.sqrt(index_array[1]**2 + self.focal_length**2)
        y = self.focal_length * index_array[1]/y
        print(x.shape, y.shape, self.images[0].shape)
        projection = cv2.remap(self.images[0][:,:,0], x, y, cv2.INTER_LINEAR)
        plt.imshow(projection[:,:,::-1])
        plt.show()
        pass

    def pairwise_alignment(self, image1, image2):
        pass

    def fix_alignment(self, panorama):
        pass

    def blending(self, panorama):
        pass

    def crop(self, panorama):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", help="feature detection method(harris, sift, msop)", default='harris' , type=str)
    parser.add_argument("--path", help="images directory path", default='./images', type=str)
    parser.add_argument("--rectangle", help="rectangling panorama(crop, warp)", default='crop', type=str)
    parser.add_argument("--focal_length", help="the focal length of images", type=float)
    args = parser.parse_args()

    images = []
    for files in os.listdir(args.path):
        images.append(cv2.imread(os.path.join(args.path,files)))
    
    stitch_instance = Stitch(images, args.detection ,args.focal_length, args.rectangle)
    stitch_instance.cylindrical()