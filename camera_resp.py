import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob 

class CameraRespFunct:
    def __init__(self, path, l, exposureTimes):
        ''' 
        Inputs: 
            path(str): path to folder with images
            l(float): smoothing factor, lambda
            exposureTimes(list): list of exposure times for image array
        '''  

        self.images = self.create_imgArr(path)                          # self.images = image array
        self.times = np.array(exposureTimes, dtype=np.float32)          # self.times = exposure times array

        self.N = len(self.images)                                       # self.N = number of images
        self.H = len(self.images[0])                                    # self.H = height of images (pixels)
        self.W = len(self.images[0][0])                                 # self.W = width of images (pixels)

        self.l = l                                                      # self.l = smoothing factor, lambda
        self.Bj = np.log(self.times)                                    # self.Bj = log of exposure times

        alignMTB = cv2.createAlignMTB()
        alignMTB.process(self.images, self.images)
    
    def create_imgArr(self, path):
        
        image_list = [] 
        
        for i, filename in enumerate(glob.glob(path + "\\" + '*.jpg')): 
            im = cv2.imread(path + '\\' + str(i) + '.jpg')
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            image_list.append(im)
        return np.array(image_list)
          
    def display_OriginalImages(self):
        fig, axs = plt.subplots(1, self.N, figsize = (20,4))
        fig.suptitle('Original Images')
        for i in range(self.N):
            axs[i].imshow(cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB))
            axs[i].axis('off')
            exp = np.round(self.times[i], 3)
            axs[i].set_title("Exposure Time = %.3f" % (exp), fontsize = 10)
        
    def w_funct(self, z):
        Zmin = 0 
        Zmax = 255 
        Zmid = (Zmax+Zmin)//2
        
        if isinstance(z, np.ndarray):
            w = np.zeros_like(z)
            w[z <= Zmid] = (z[z <= Zmid] - Zmin + 1)
            w[z > Zmid] = (Zmax - z[z > Zmid] + 1)
            return w

        else:
            if z <= Zmid:
                return  (z - Zmin + 1)
            else: 
                return  (Zmax - z + 1)

 
        
    def getSamples(self):
    
        numPixels = self.H * self.W
        numSamples = 50

        stepSize = int(np.floor(numPixels / numSamples))

        samplePixels = np.arange(0,numPixels,stepSize)[:numSamples]
        
        # get flatten image 
        # self.flattenedImages = np.array(self.images.reshape(self.N, 3, numPixels),dtype=np.uint8)
        self.flattenedImages = np.zeros((self.N, 3, numPixels),dtype=np.uint8)
        for n in range(self.N):
            for c in range(3):
                self.flattenedImages[n,c] = self.images[n][:,:,c].flatten()

        # get the sampled pixel values 
        self.ZR = self.flattenedImages[:,2,samplePixels].T
        self.ZG = self.flattenedImages[:,1,samplePixels].T
        self.ZB = self.flattenedImages[:,0,samplePixels].T


    def gsolve(self,Z):
        # this function is a python version of the one provided by Debevec's paper. 
        '''        
        solve for imaging system response function 

        Given a set of pixel values observed for several pixels 
        in several images with different exposure times, 

        this funcction returns the imaging system's response function g as well as the log film irradiance values 
        for the observed pixels 


        Inputs: 
        Z(i,j) : pixel values of pixel locations number i in image j 
        B(j)   : the log delta t for image j
        l   : lambda, the constant that determines the amount of smoothness
        w(z): the weighting function value for pixel value z 

        outputs:
        g(z) : log exposure corresponding to pixel value z 
        lE(i): the log film irradiance at pixel location i 
        '''

        n = 256

        s1, s2 = Z.shape
        A = np.zeros((s1*s2+n+1, n+s1))
        b = np.zeros((A.shape[0],1))

        ## include the data-fitting equations 
        k=0
        for i in range(s1):
            for j in range(s2):
                wij = self.w_funct(Z[i,j])
                A[k,Z[i,j]] = wij
                A[k,n+i] = -wij
                b[k] = wij*self.Bj[j]
                k +=1

        # fix the curve by setting its middle value to zero     
        A[k,129] = 0
        k +=1
    
        # include the smoothness equations 
        for i in range(1,n-2):
            A[k,i] = self.l*self.w_funct(i+1)
            A[k,i+1] = -2*self.l*self.w_funct(i+1)
            A[k,i+2] = self.l*self.w_funct(i+1)
            k +=1
        
        # solve the system using Singular Value Decomposition(SVD)
        x = np.linalg.lstsq(A,b,rcond=None) 
        
        x = x[0]
        g = x[:n].flatten()
        lE = x[n:].flatten()    
        
        return g, lE
    
    def getCameraResp(self):

        self.getSamples()
        
        self.gR, self.lER = self.gsolve(self.ZR)
        self.gG, self.lEG = self.gsolve(self.ZG)
        self.gB, self.lEB = self.gsolve(self.ZB)

        gList = [self.gR, self.gG, self.gB]
        lEList = [self.lER, self.lEG, self.lEB]

        return gList, lEList
    
    def recover_HDR_RadianceMap(self):
        m = np.zeros((self.flattenedImages.shape[1:]))
        wsum = np.zeros(self.flattenedImages.shape[1:])
        hdr = np.zeros(self.flattenedImages.shape[1:])
        
        lnDt = np.log(self.times)

        for i in range(self.N):

            wij_B = self.w_funct(self.flattenedImages[i,2])
            wij_G = self.w_funct(self.flattenedImages[i,1])
            wij_R = self.w_funct(self.flattenedImages[i,0])
            
            wsum[0,:] += wij_B
            wsum[1,:] += wij_G
            wsum[2,:] += wij_R
            
            m2 = np.subtract(self.gB[self.flattenedImages[i,0]],lnDt[i])
            m1 = np.subtract(self.gG[self.flattenedImages[i,1]],lnDt[i])
            m0 = np.subtract(self.gR[self.flattenedImages[i,2]],lnDt[i])
    
            hdr[0] += np.multiply(m0,wij_B)
            hdr[1] += np.multiply(m1,wij_G)
            hdr[2] += np.multiply(m2,wij_R)

        hdr = np.divide(hdr,wsum)
        hdr = np.exp(hdr)
        hdr = np.reshape(np.transpose(hdr), (self.H,self.W,3))
                    
        self.imgf32 = (hdr/np.amax(hdr)*255).astype(np.float32)
        plt.figure(constrained_layout=False,figsize=(10,10))
        plt.title("fused HDR radiance map", fontsize=20)
        plt.imshow(self.imgf32)  

    def plotResponseCurves(self, axs, r, c): 

        px = list(range(0,256))
        axs[r,c].set_title(f"Inverse Response Curve g(Z_ij), lambda = {self.l}")
        axs[r,c].plot(px,np.exp(self.gR),'r')
        axs[r,c].plot(px,np.exp(self.gB),'b')
        axs[r,c].plot(px,np.exp(self.gG),'g')
        axs[r,c].set_ylabel("log Exposure X")
        axs[r,c].set_xlabel("Pixel value Z")      

if __name__ == "__main__":
    path = r"C:\Users\audre\OneDrive\Documents\GitHub\EE367_Final_Project\new_images"
    lambdas = np.array([0.01, 0.1, 1, 10, 1000,10000])
    exposures = [1/1, 1/5, 1/13, 1/25, 1/60, 1/100]
    fig, axs = plt.subplots(2,3, figsize = (20, 10))
    for i, l in enumerate(lambdas):
        crf = CameraRespFunct(path, l, exposures)
        g_list, lE_list = crf.getCameraResp()
        r = i // 3
        c = i % 3
        crf.plotResponseCurves(axs, r, c)
    plt.show()