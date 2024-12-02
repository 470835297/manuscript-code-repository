import os
from PIL import Image
import numpy as np
import cv2
import pylab
import matplotlib.pyplot as plt 
img_path = 'test_output/cell_100/'
file_img = sorted(os.listdir(img_path))
label_path = 'dataset/cell/testlabel/'

file_label = sorted(os.listdir(label_path))
picture_len = len(file_img)
label_len = len(file_label)
pictures=np.zeros((640,800,picture_len))
labels=np.zeros((640,800,label_len))
#for step in range(len(file_img)):
for step in range(picture_len):
    img_name,_=os.path.splitext(os.path.basename(file_img[step]))
    label_name,_=os.path.splitext(os.path.basename(file_label[step]))
    img_paths=img_path+img_name+'.png'
    #img=np.array(image_name)
    label_paths=label_path+label_name+'.jpg'
    picture=cv2.imread(img_paths,2)
    label=cv2.imread(label_paths,2)
    #plt.imshow(picture)
    #plt.show()
    #plt.imshow(label)
    #plt.show()
    #picture = cv2.resize(picture, (256, 256),interpolation=cv2.INTER_CUBIC)
    
    picture=picture/255.0
    #picture[picture==1.0]=1
    #picture[picture<1.0]=0
    label=label/255.0
    #label[label==1.0]=1
    #label[label<1.0]=0
     
    
    pictures[:,:,step]=picture
    labels[:,:,step]=label
#pictures = np.swapaxes(pictures,0,1)
#labels = np.swapaxes(labels,0,1) 





tp = np.sum(np.multiply(pictures,labels))
tn = np.sum(np.multiply((1-pictures),(1-labels)))

fp = np.sum(np.multiply(pictures,(1-labels)))
fn = np.sum(np.multiply((1-pictures),labels))
print('Dice:%.4f' % (2*tp/(2*tp+fp+fn)))
print('sensitivity:%.4f' %(tp/(tp+fn)))
print('specificity:%.4f' %(tn/(tn+fp)))
print('Accuracy:%.4f' %((tn+tp)/(tn+fp+fn+tp)))



'''numsample=256
pictures_split = np.zeros((124,256,numsample))
labels_split = np.zeros((124,256,numsample))
for step in range (0,10):
    pictures_split=pictures[:,:,step*numsample:(step+1)*numsample]
    labels_split=labels[:,:,step*numsample:(step+1)*numsample]
    
    tp = np.sum(np.multiply(pictures_split,labels_split))
    tn = np.sum(np.multiply((1-pictures_split),(1-labels_split)))

    fp = np.sum(np.multiply(pictures_split,(1-labels_split)))
    fn = np.sum(np.multiply((1-pictures_split),labels_split))
    print(2*tp/(2*tp+fp+fn))
    print(tp/(tp+fn))
    print(tn/(tn+fp))
    print(step)'''






    


    