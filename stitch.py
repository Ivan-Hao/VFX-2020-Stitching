import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import os
import argparse

class Stitch():

    def __init__(self, images, detection, focal, rectangle):
        self.images = images
        self.detection = detection
        self.featrue = None
        self.focal = focal
        self.rectangle = rectangle
        self.panorama = None

    def feature_detect(self, k=0.04):
        for i in range(len(self.images)):
            img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
            img = cv2.blur(img, (3, 3))
            Iy,Ix = np.gradient(img)
            Ixx = Ix*Ix
            Iyy = Iy*Iy
            Ixy = Ix*Iy
            Sx = cv2.blur(Ixx, (3, 3))
            Sy = cv2.blur(Iyy, (3, 3))
            Sxy = cv2.blur(Ixy, (3, 3))
            detM = Sx*Sy - Sxy*Sxy
            traceM = Sx+Sy
            R = detM - k*(traceM * traceM)
            plt.imshow(R, cmap='jet')
            plt.show()
            #local maximum 還沒有做

        

        pass

    def feature_matching(self, feature):
        pass

    def cylindrical(self):

        for i in range(len(self.images)):
            shape = self.images[i].shape[0:2]
            index = np.indices(shape)
            h = shape[0]
            w = shape[1]
            x_shift = w/2
            y_shift = h/2
            x = index[1] - x_shift
            y = index[0] - y_shift
            x_prime = self.focal[i] * np.arctan(x / self.focal[i])
            y_prime = self.focal[i] * y / np.sqrt(x**2 + self.focal[i]**2)
            x_prime += x_shift
            y_prime += y_shift
            x_prime = np.round(x_prime).astype(np.uint)
            y_prime = np.round(y_prime).astype(np.uint)
            project = np.zeros_like(self.images[i])
            project[y_prime,x_prime,:] = self.images[i][index[0],index[1],:]
            plt.imshow(project[:,:,::-1])
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
    parser.add_argument("--focal", help="the focal length file", default='./focal.txt', type=str)
    args = parser.parse_args()

    images = []
    for files in os.listdir(args.path):
        images.append(cv2.imread(os.path.join(args.path,files)))
    focal = np.loadtxt(args.focal)
    stitch_instance = Stitch(images, args.detection , focal, args.rectangle)
    #stitch_instance.cylindrical()
    stitch_instance.feature_detect()