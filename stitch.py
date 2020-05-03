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
        self.feature_position = [] # 每張圖每個feature 的位置 harris only
        self.match = [] # i 與 i+1張圖之間的match match有(distance , index1, index2), index1是第i張圖第index1個feature
        self.feature = [] # 全部的image feature
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
                self.feature_descriptor(img,np.where(R != 0))
        else:
            sift = cv2.xfeatures2d.SIFT_create(100)
            for i in range(len(self.images)):
                img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
                self.gray_images.append(img)
                self.feature.append(sift.detectAndCompute(self.gray_images[i],None)) # (key, 128 vector)
            
            
                    
    def feature_descriptor(self, image, position):
        if self.detection == 'harris':     
            self.feature_position.append(position)
            des = []
            for i in range(len(position[0])):
                f = image[position[0][i]-2:position[0][i]+3,position[1][i]-2:position[1][i]+3].flatten().astype(np.float64)
                des.append(f)
            self.feature.append(np.array(des)) # (25 vector)
                
    

    def feature_matching(self):
        if self.detection == 'harris':
            for i in range(len(self.feature)-1):
                tree = spatial.KDTree(self.feature[i])
                match = []
                for j in range(len(self.feature[i+1])):
                    distance,index = tree.query(self.feature[i+1][j],2)
                    if index[0] < 0.5*index[1]:
                        match.append((distance[0],index[0],j)) # (distance, feature index, feature index)
                match = sorted(match, key = lambda s: s[0])
                self.match.append(match)
                '''
                # 拼接看看
                for j in range(10):
                    temp = np.concatenate((self.gray_images[i],self.gray_images[i+1]),axis=1)
                    plt.plot(self.feature_position[i][1][match[j][1]],self.feature_position[i][0][match[j][1]],'-r*')
                    plt.plot(384+self.feature_position[i+1][1][match[j][2]],self.feature_position[i+1][0][match[j][2]],'-r*')
                plt.imshow(temp,cmap='gray')
                plt.show()
                '''
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            for i in range(len(self.images)-1):
                kp1,des1 = self.feature[i]
                kp2,des2 = self.feature[i+1]
                    
                matches = flann.knnMatch(des1, des2, k=2)
                match = []
                for m, n in matches:
                    if m.distance < 0.50*n.distance:
                        match.append((kp1[m.queryIdx].pt,kp2[m.trainIdx].pt))
                self.match.append(match) #只有match沒有 feature position ((x1,y1) , (x2,y2))
                '''
                for j in range(10):
                    temp = np.concatenate((self.gray_images[i],self.gray_images[i+1]),axis=1)
                    plt.plot(match[j][0][0],match[j][0][1],'-r*')
                    plt.plot(384+match[j][1][0],match[j][1][1],'-r*')
                plt.imshow(temp,cmap='gray')
                plt.show()
                '''
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
        panorama = np.zeros((self.images[0].shape[0],self.images[0].shape[1]*len(self.images)))
        if self.detection == 'harris':
            A = np.zeros((20,6))
            b = np.zeros((20,1))
            for i in range(len(self.images)-1):
                for j in range(10):
                    # origin x                                       # origin y
                    b[j*2,0] = self.feature_position[i][1][self.match[i][j][1]]
                    b[j*2+1] = self.feature_position[i][0][self.match[i][j][1]]
                    # match x                                             # match y
                    A[j*2,0:3] = np.array([self.feature_position[i+1][1][self.match[i][j][2]], self.feature_position[i+1][0][self.match[i][j][2]],1])
                    A[j*2+1,3:6] = np.array([self.feature_position[i+1][1][self.match[i][j][2]], self.feature_position[i+1][0][self.match[i][j][2]],1])
                A_inverse = np.linalg.pinv(A)
                motion_model = np.dot(A_inverse,b).reshape(2,3)
                rows,cols = self.images[i+1].shape[:2]
                result = cv2.warpAffine(self.images[i+1],motion_model,(rows,cols))
                plt.imshow(result[:,:,::-1])
                plt.show()
        else:
            for i in range(len(self.images)-1):
                A = np.zeros((20,6))
                b = np.zeros((20,1))
                for j in range(10):
                    print(len(self.match),len(self.match[0]),len(self.match[0][0]))
                    b[j*2,0] = self.match[i][j][0][0] # origin x
                    b[j*2+1] = self.match[i][j][0][1] # origin y
                    A[j*2,0:3] = np.array([self.match[i][j][1][0],self.match[i][j][1][1],1]) # x_prime
                    A[j*2+1,3:6] = np.array([self.match[i][j][1][0],self.match[i][j][1][1],1]) # y_prime
                print(A)
                print(b)
                A_inverse = np.linalg.pinv(A)
                motion_model = np.dot(A_inverse,b).reshape(2,3)
                print(motion_model)
                rows,cols = self.images[i+1].shape[:2]
                result = cv2.warpAffine(self.images[i+1],motion_model,(cols,rows))
                plt.imshow(result[:,:,::-1])
                plt.show()
                break

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
    stitch_instance.feature_matching()
    stitch_instance.pairwise_alignment()