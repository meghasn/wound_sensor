import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process(ref,img,outp,path,name):
     #register
     register(ref,img,outp,path,name)

#perform image registration and add result to the above list
def register(img1,img2,outp,path,name):
    
    
    akaze = cv2.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
            
    # Draw matches
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(path+'/'+'matches_height.jpg', img3)



    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

    # Compute homography
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,8.0)
    print(H.shape)
    # Warp image
    warped_image = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
                
    cv2.imwrite(path+'/'+'warped_height.jpg', warped_image) 


    # warped_img=cv2.imread('warped_normal.jpg')
    outp.append(warped_image)
    masked_img1=create_mask(img1)
    masked_img2=create_mask(warped_image)
    roi_img1=extract(img1,masked_img1)
    
    roi_img2=extract(warped_image,masked_img2)
    
    diff_img=difference(roi_img1,roi_img2)
    
    # diff_img=cv2.imread('diff.jpg')
    overlay(img1,diff_img,path,name)
    
    # cv2.imwrite('overlay.jpg',overlayed)



def create_mask(image):
     # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the yellow color in HSV
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # Create a mask for the yellow color using the lower and upper bounds
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    return yellow_mask

def extract(image,mask):
     # Apply the mask to the image using bitwise AND operation
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Extract the region of interest
    roi = masked_image

    return roi

def difference(image1,image2):
     print(image1,image2)
     diff = cv2.absdiff(image1, image2)
     cv2.imwrite('diff.jpg',diff)
     outp.append(diff)
     return diff

def overlay(original,diff,path,name):
    # Apply the colormap to the heatmap
    # cv2.imwrite('diff2.jpg',diff)
    # # Convert the original image to grayscale
    # heatmap_img = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    

    # # Overlay the original image and the heatmap
    # overlay = cv2.addWeighted(original, 0.5, heatmap_img, 0.5, 0)
    # cbar = fig.colorbar(overlay)
    # return overlay
    # Convert the original image to grayscale
    fig, ax = plt.subplots()
    heatmap_img= cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    # Overlay the original image and the heatmap
    overlay = cv2.addWeighted(original, 0.5, heatmap_img, 0.5, 0)
    cv2.imwrite('original.jpg',overlay)
    # Plot the overlay image
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.colorbar()  # Add the colorbar
    print(path)
    # Save the figure with the colorbar
    plt.savefig(path+'/'+'overlay.jpg')
    overlay=cv2.imread(path+'/'+'overlay.jpg')
    outp.append(overlay)
    print(len(outp))
    
    print_result(outp,name,path)

def print_result(images,name,path):
    headings=['reference',name,'registered_image','difference_image','result']
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))

    for i in range(num_images):
        ax = axes[i]
        image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(headings[i])
        ax.axis('off')

    plt.savefig(path+'/'+'result.jpg', bbox_inches='tight')
    plt.pause(.5)

     
     

    

     
     
     



print("megha")
#read input
outp=[]
print(outp)
category=os.listdir('stand images/0mM')
# process('stand images\10mM\Dry\S1(10mM)Dry.jpg','stand images\10mM\Android\S1(10mM)android.jpg',outp,'output\10mM\Android\1','str(i)++reference_li[j]')
for i in category:
    if i=='Thumbs.db' or i=='Dry' or i=='.DS_Store':
            continue
    samples=sorted(os.listdir('stand images/0mM/'+str(i)))
    os.mkdir('output/0mM/'+str(i))
    for j in range(len(samples)):
        fig, ax = plt.subplots()
        outp=[]
        os.mkdir('output/0mM/'+str(i)+'/'+str(j+1))
        path='output/0mM/'+str(i)+'/'+str(j+1)
        reference_li=sorted(os.listdir('stand images/0mM/Dry'))
        reference_img=cv2.imread('stand images/0mM/Dry/'+reference_li[j])
        outp.append(reference_img)
        print('stand images/0mM/Dry/'+reference_li[j])
        img=cv2.imread('stand images/0mM/'+str(i)+'/'+samples[j])
        outp.append(img)
        print('stand images/0mM/'+str(i)+'/'+samples[j])
        process(reference_img,img,outp,path,str(i)+'_'+reference_li[j])#resultant image
    
    
        
        
    
        
       
        
        

