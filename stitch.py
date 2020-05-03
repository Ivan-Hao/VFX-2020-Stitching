import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import spatial
import os
import argparse

class Stitch():

    def __init__(self, images, detection ,focal, rectangle):
        self.images = images # bgr image list
        self.detection = detection
        self.gray_images = [] # gray image list
        self.featrue_position = [] # 每張圖每個feature 的位置
        self.match = [] # i 與 i+1張圖之間的match match有(distance , index1, index2), index1是第i張圖第index1個feature
        self.featrue = [] # 全部的image feature
        self.focal = focal # list of focal length
        self.rectangle = rectangle
        self.panorama = None

    def feature_detect(self, k=0.04):
        if self.detection == 'harris':

            for i in range(len(self.images)):
                #harris corner detection
                img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
                self.gray_images.append(img)
                img = np.float64(img) #一定要轉= = 
                img = cv2.GaussianBlur(img, (3, 3),0)
                Iy,Ix = np.gradient(img)
                Ixx = Ix*Ix
                Iyy = Iy*Iy
                Ixy = Ix*Iy
                Sx = cv2.GaussianBlur(Ixx, (3, 3),0)
                Sy = cv2.GaussianBlur(Iyy, (3, 3),0)
                Sxy = cv2.GaussianBlur(Ixy, (3, 3),0)
                detM = Sx*Sy - Sxy*Sxy
                traceM = Sx+Sy
                R = detM - k*(traceM * traceM)
                
                R[:10,:]=0; R[-10:,:]=0; R[:,:10]=0; R[:,-10:]=0 # 去掉邊邊的
                threshold = np.percentile(R, ((1 - 512/(img.shape[0]*img.shape[1]))*100))
                R[np.where(R<threshold)] = 0
                #non maximum suppression
                local_max = filters.maximum_filter(R, (5, 5))
                R[np.where(R != local_max)] = 0
                '''
                #==================
                plt.plot(np.where(R != 0)[1],np.where(R != 0)[0],'r*')
                plt.imshow(R,cmap='gray')
                plt.show()
                break
                #==================
                '''
                # feature descriptor
                self.featrue_descriptor(img,np.where(R != 0))
        else:
            sift = cv2.xfeatures2d.SIFT_create()
            for i in range(len(self.images)):
                img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
                self.gray_images.append(img)
            psd_kp1, psd_des1 = sift.detectAndCompute(self.gray_images[0],None)
            psd_kp2, psd_des2 = sift.detectAndCompute(self.gray_images[1],None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(psd_des1, psd_des2, k=2)
            goodMatch = []
            for m, n in matches:
                if m.distance < 0.50*n.distance:
                    goodMatch.append(m)
                    #print(psd_kp1[m.queryIdx].pt,psd_kp2[m.trainIdx].pt)
                    
            # 增加一个维度
            goodMatch = np.expand_dims(goodMatch, 1)
            print(goodMatch.shape)

            
    def featrue_descriptor(self, image, position):
        self.featrue_position.append(position)
        des = []
        for i in range(len(position[0])):
            f = image[position[0][i]-2:position[0][i]+3,position[1][i]-2:position[1][i]+3].flatten().astype(np.float64)
            des.append(f)
        self.featrue.append(np.array(des))
                
    

    def feature_matching(self):
        for i in range(len(self.featrue)-1):
            tree = spatial.KDTree(self.featrue[i])
            match = []
            for j in range(len(self.featrue[i+1])):
                distance,index = tree.query(self.featrue[i+1][j],2)
                if index[0] < 0.5*index[1]:
                    match.append((distance[0],index[0],j))
            match = sorted(match, key = lambda s: s[0])
            self.match.append(match)

            # 拼接看看
            for j in range(10):
                temp = np.concatenate((self.gray_images[i],self.gray_images[i+1]),axis=1)
                plt.plot(self.featrue_position[i][1][match[j][1]],self.featrue_position[i][0][match[j][1]],'-r*')
                plt.plot(384+self.featrue_position[i+1][1][match[j][2]],self.featrue_position[i+1][0][match[j][2]],'-r*')
            plt.imshow(temp,cmap='gray')
            plt.show()
                     
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
            x_prime = np.around(x_prime).astype(np.uint)
            y_prime = np.around(y_prime).astype(np.uint)
            project = np.zeros_like(self.images[i])
            project[y_prime,x_prime,:] = self.images[i][index[0],index[1],:]
            self.images[i] = project
            '''
            #==============
            plt.imshow(project[:,:,::-1])
            plt.show()
            #==============
            '''        
        pass

    def pairwise_alignment(self):
        A = np.zeros((20,6))
        b = np.zeros((20,1))
        for i in range(len(self.images)-1):
            for j in range(10):
                # origin x                                       # origin y
                b[j*2,0] = self.featrue_position[i][1][self.match[i][j][1]]
                b[j*2+1] = self.featrue_position[i][0][self.match[i][j][1]]
                # match x                                             # match y
                A[j*2,0:3] = np.array([self.featrue_position[i+1][1][self.match[i][j][2]], self.featrue_position[i+1][0][self.match[i][j][2]],1])
                A[j*2+1,3:6] = np.array([self.featrue_position[i+1][1][self.match[i][j][2]], self.featrue_position[i+1][0][self.match[i][j][2]],1])
            A_inverse = np.linalg.pinv(A)
            motion_model = np.dot(A_inverse,b).reshape(2,3)
            rows,cols = self.images[i+1].shape[:2]
            result = cv2.warpAffine(self.images[i+1],motion_model,(rows,cols))
            plt.imshow(result[:,:,::-1])
            plt.show()
            
        pass

    def fix_alignment(self, panorama):
        pass

    def blending(self, panorama):
        pass

    def crop(self, panorama):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", help="feature detection method(harris, sift)", default='sift', type=str)
    parser.add_argument("--path", help="images directory path", default='./images', type=str)
    parser.add_argument("--rectangle", help="rectangling panorama(crop, warp)", default='crop', type=str)
    parser.add_argument("--focal", help="the focal length file", default='./focal.txt', type=str)
    args = parser.parse_args()

    images = []
    for files in os.listdir(args.path):
        images.append(cv2.imread(os.path.join(args.path,files)))
    focal = np.loadtxt(args.focal)
    stitch_instance = Stitch(images ,args.detection ,focal, args.rectangle)
    #stitch_instance.cylindrical()
    stitch_instance.feature_detect()
    #stitch_instance.feature_matching()
    #stitch_instance.pairwise_alignment()