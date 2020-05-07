import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.spatial import KDTree
import os
import argparse
from des_sift import SIFTDescriptor 

class Stitch():

    def __init__(self, images, detection ,descriptor ,focal):
        self.images = images # bgr image list
        self.gray_images = [] # gray image list
        self.detection = detection
        self.descriptor = descriptor
        self.feature_position = [] # 每張圖每個feature 的位置 harris only
        self.match = [] # i 與 i+1張圖之間的match match有 ((x,y), (x_prime,y_prime)) 
        self.feature = [] # 全部的image feature
        self.focal = focal # list of focal length
        self.motion = [] # i 與 i+1 motion model
        self.panorama = None
        self.up_right = None
        self.down_right = None

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
            project = project[np.min(y_prime):np.max(y_prime),np.min(x_prime):np.max(x_prime)]
            self.images[i] = project.astype(np.uint8)

    def feature_detect(self, k=0.04):
        if self.detection == 'harris':
            for i in range(len(self.images)):
                # =================== harris corner detection ===================================== #
                img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
                self.gray_images.append(img)
                img = img.astype(np.float64) #一定要轉= = 
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
                R[:10,:]=0; R[-10:,:]=0; R[:,:10]=0; R[:,-10:]=0 # 去掉邊邊的特徵點
                threshold = np.percentile(R, ((1 - 512/(img.shape[0]*img.shape[1]))*100))
                R[np.where(R<threshold)] = 0
                # ======================= non maximal suppression =================================== #
                local_max = filters.maximum_filter(R, (7, 7))
                R[np.where(R != local_max)] = 0
                # ======================== feature descriptor ======================================= #
                self.feature_descriptor(img, np.where(R != 0))
                '''
                plt.plot(np.where(R != 0)[1],np.where(R != 0)[0],'r*')
                plt.imshow(R,cmap='gray')
                plt.show()
                '''
        else:
            # ============================ sift detection and description ================================ #
            sift = cv2.xfeatures2d.SIFT_create()
            for i in range(len(self.images)):
                img = cv2.cvtColor(self.images[i],cv2.COLOR_BGR2GRAY)
                self.gray_images.append(img)
                self.feature.append(sift.detectAndCompute(self.gray_images[i],None)) # (key, 128 vector)
                                
    def feature_descriptor(self, image, position): # harris only  
        self.feature_position.append(position)
        des = []
        if self.descriptor == 'patch':
            for i in range(len(position[0])):
                des.append(image[position[0][i]-6:position[0][i]+6,position[1][i]-6:position[1][i]+6].flatten())
        else:
            SD = SIFTDescriptor(patchSize = 16) 
            for i in range(len(position[0])):
                patch = image[position[0][i]-8:position[0][i]+8,position[1][i]-8:position[1][i]+8]
                sift = SD.describe(patch) 
                des.append(sift)
        self.feature.append(np.array(des)) # (121 or 128 vector)

    def feature_matching(self):
        if self.detection == 'harris':
            for i in range(len(self.feature)-1):
                tree = KDTree(self.feature[i])
                m = [] 
                for j in range(len(self.feature[i+1])):
                    distance,index = tree.query(self.feature[i+1][j],2)
                    if distance[0] < 0.5*distance[1]:
                        m.append(((self.feature_position[i][1][index[0]],self.feature_position[i][0][index[0]]),(self.feature_position[i+1][1][j],self.feature_position[i+1][0][j])))    
                self.match.append(m)
                '''
                for j in range(10):
                    temp = np.concatenate((self.gray_images[i],self.gray_images[i+1]),axis=1)
                    r = np.random.rand()
                    b = np.random.rand()
                    g = np.random.rand()
                    plt.plot(m[j][0][0],m[j][0][1],'*',color= (r,g,b))
                    plt.plot(self.images[i].shape[1]+m[j][1][0],m[j][1][1],'*',color = (r,g,b))
                plt.imshow(temp,cmap='gray')
                plt.show()
                '''             
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            for i in range(len(self.images)-1):
                kp1,des1 = self.feature[i]
                kp2,des2 = self.feature[i+1]
                # ===================交叉比對=================== #      
                matches1 = flann.knnMatch(des1, des2, k=2)
                matches2 = flann.knnMatch(des2, des1, k=1)
                qid = [i[0].queryIdx for i in matches2]
                tid = [i[0].trainIdx for i in matches2]
                def fn(x):
                    if x[0].queryIdx in tid and x[0].trainIdx in qid:
                        if qid.index(x[0].trainIdx) == tid.index(x[0].queryIdx):
                            return x
                    return None
                matches = list(filter(fn,matches1))
                m_ = []
                for m, n in matches:
                    if m.distance < 0.4*n.distance:
                        m_.append((kp1[m.queryIdx].pt,kp2[m.trainIdx].pt))
                self.match.append(m_) #((x1,y1) , (x2,y2))
                '''
                for j in range(10):
                    temp = np.concatenate((self.gray_images[i],self.gray_images[i+1]),axis=1)
                    r = np.random.rand()
                    b = np.random.rand()
                    g = np.random.rand()
                    plt.plot(m[j][0][0],m[j][0][1],'*',color= (r,g,b))
                    plt.plot(self.images[i].shape[1]+m[j][1][0],m[j][1][1],'*',color = (r,g,b))
                plt.imshow(temp,cmap='gray')
                plt.show()
                '''
    def pairwise_alignment(self):
        for i in range(len(self.match)):
            # ==========================RANSAC================================ #
            origin_mat = np.array(self.match[i]).flatten().reshape(-1,2)[0::2,:].T # 第 i 張 key position (x,y)
            kp_mat = np.array(self.match[i]).flatten().reshape(-1,2)[1::2,:].T # 第 i+1 張 key position (x,y)
            kp_mat = np.vstack((kp_mat,np.ones((1,kp_mat.shape[1])))) # 增加第三維
            best_motion = None
            min_ = np.inf
            for k in range(500): # 200次啦 = =
                A = np.zeros((6,6),dtype=np.float64)
                b = np.zeros((6,1),dtype=np.float64)
                for j in range(3):
                    x = np.random.randint(len(self.match[i]))
                    b[j*2,0] = self.match[i][x][0][0] # origin x   
                    b[j*2+1] = self.match[i][x][0][1] # origin y
                    A[j*2,0:3] = np.array([self.match[i][x][1][0],self.match[i][x][1][1],1]) # x_prime, y_prime, 1 ,0 ,0 ,0
                    A[j*2+1,3:6] = np.array([self.match[i][x][1][0],self.match[i][x][1][1],1]) # 0, 0, 0, x_prime, y_prime, 1
                A_inverse = np.linalg.pinv(A)
                motion_model = np.dot(A_inverse,b).reshape(2,3)
                motion_model = np.vstack((motion_model,[0,0,1]))
                compare = np.dot(motion_model,kp_mat)
                diff = np.abs(compare[:2,:].flatten('F') - origin_mat.flatten('F')).sum()
                if diff < min_:
                    min_ = diff
                    best_motion = motion_model
            self.motion.append(best_motion)

    def image_matching(self):
            last = self.images[0]
            m = self.motion[0]
            for i in range(len(self.motion)):
                img = self.images[i+1]
                pos1 = np.dot(m,np.array([img.shape[1]-1,img.shape[0]-1,1]).reshape(3,1)) #右下座標
                pos2 = np.dot(m,np.array([0,img.shape[0]-1,1]).reshape(3,1)) #左上
                pos3 = np.dot(m,np.array([0,0,1]).reshape(3,1)) #左上
                pos4 = np.dot(m,np.array([img.shape[1]-1,0,1]).reshape(3,1)) #右下
                max_x = max(pos1[0,0],pos2[0,0],pos3[0,0],pos4[0,0])
                max_y = max(pos1[1,0],pos2[1,0],pos3[1,0],pos4[1,0])
                pos = np.array([max_x,max_y])
                min_y = min(pos1[1,0],pos2[1,0],pos3[1,0],pos4[1,0])
                if min_y < 0:
                    last = np.pad(last, ((int(abs(min_y)+1),0),(0,0),(0,0))) 
                    plt.imshow(last.astype(np.uint8))
                    plt.show()
                panorama = cv2.warpPerspective(img,m,(int(pos[0]), int(pos[1]+abs(min_y)))).astype(np.float64)
                # ===================================blending==================================== #
                blend_mat = panorama.sum(axis=2).astype(np.bool)
                y_shift = blend_mat.shape[1]-last.shape[1]
                x_shift = blend_mat.shape[0]-last.shape[0] 
                if min_y >= 0 :
                    kk = np.pad(last.sum(axis=2), ((0,x_shift),(0,y_shift))).astype(np.bool)
                else :
                    kk = np.pad(last.sum(axis=2), ((0,x_shift),(y_shift,0))).astype(np.bool)           
                blend_mat = np.logical_and(blend_mat,kk)   
                r = blend_mat.copy().astype(np.float64)
                l = blend_mat.copy().astype(np.float64)
                for j in range(blend_mat.shape[0]):
                    if blend_mat[j].sum() !=0 :
                        nonzero = np.nonzero(blend_mat[j])
                        min_ = np.min(nonzero)
                        max_ = np.max(nonzero)
                        t = max_ - min_
                        l[j][min_:max_+1] = np.linspace(1,0,t+1,dtype=np.float64)
                        r[j][min_:max_+1] = np.linspace(0,1,t+1,dtype=np.float64)
                last = np.pad(last, ((0,panorama.shape[0]-last.shape[0]),(0,panorama.shape[1]-last.shape[1]),(0,0)))
                for j in range(3):
                    r_ = r*panorama[:,:,j]
                    l_ = l*last[:,:,j]
                    panorama[:,:,j] += last[:,:,j]
                    panorama[:,:,j] *= (blend_mat^True) 
                    panorama[:,:,j] += r_
                    panorama[:,:,j] += l_
                # ================================================================================= #
                last = panorama
                if i == len(self.motion)-1:
                    up_right = np.array([self.images[i+1].shape[1]-1,0,1]).reshape(3,1)
                    up_right = np.dot(m,up_right)
                    self.up_right = up_right
                    self.down_right = pos
                    break
                m = np.dot(m,self.motion[i+1])
            self.panorama = last
            plt.imshow(self.panorama[:,:,::-1].astype(np.uint8))
            plt.show()

    def fix_alignment(self):
        src = self.panorama
        srcTri = np.array( [[0, 0], [0, self.images[0].shape[0] - 1], [self.up_right[0,0],self.up_right[1,0]], [self.down_right[0],self.down_right[1]]]).astype(np.float32)
        dstTri = np.array( [[0, 0], [0, self.images[0].shape[0] - 1], [self.up_right[0,0], 0], [self.up_right[0,0],self.images[0].shape[0] - 1]]).astype(np.float32)
        warp_mat = cv2.getPerspectiveTransform(srcTri, dstTri)
        warp_dst = cv2.warpPerspective(src, warp_mat, ( int(self.up_right[0,0]), self.images[0].shape[0] - 1))
        self.panorama = warp_dst

    def crop(self):
        self.panorama = self.panorama[10:,:,:]
        p = self.panorama.sum(axis= 2)
        index = np.where(p==0)
        m = np.min(index[0])
        self.panorama = self.panorama[:m,:,:]
        plt.imshow(self.panorama[:,:,::-1])
        plt.show()     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", help="feature detection method(harris, sift)", default='sift', type=str)
    parser.add_argument("--descriptor", help="feature descriptor method(patch, sift)", default='sift', type=str)
    parser.add_argument("--path", help="images directory path", default='./images', type=str)
    parser.add_argument("--warp", help="warp (affine, transform)", default='affine', type=str)
    parser.add_argument("--focal", help="the focal length file", default='./focal.txt', type=str)
    args = parser.parse_args()

    images = []
    for files in os.listdir(args.path):
        images.append(cv2.imread(os.path.join(args.path,files)))
    focal = np.loadtxt(args.focal)
    stitch_instance = Stitch(images ,args.detection ,args.descriptor ,focal)
    stitch_instance.cylindrical()
    stitch_instance.feature_detect()
    stitch_instance.feature_matching()
    stitch_instance.pairwise_alignment()
    stitch_instance.image_matching()
    stitch_instance.fix_alignment()
    stitch_instance.crop()