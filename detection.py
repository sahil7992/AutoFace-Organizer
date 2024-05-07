import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import cv2  as cv
from timeit import default_timer as timer
import pickle


start=timer()


# Check if CUDA is available
if torch.cuda.is_available():

    device = torch.device("cuda")
else:

    device = torch.device("cpu")

#loading datsets path

folder_path = "path for images you want to do detction"

with open('path to model stored', 'rb') as f:
    data = pickle.load(f)
known_embeddings, name= data

#intializing model and MTCNN for face recognition
resnet = InceptionResnetV1(pretrained='vggface2',device=device).eval()
mtcnn = MTCNN(keep_all=True,device=device)


#function for getting embeddings form source data images for each face  
def get_embeddings(img_path):
    img = cv.imread(img_path)
    img1=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    faces= mtcnn(img1)
    if faces is None:

        return None,None
    num_faces = len(faces)    
    boxes, _ = mtcnn.detect(img1)
    print(_)
    embeddings = []
    for face in faces:
        emb = resnet(face.unsqueeze(0)).detach().numpy()[0]
        embeddings.append(emb)
    return embeddings,boxes       


#creating folder to store matched data
matched_folder = 'matched_images'
if not os.path.exists(matched_folder):
    os.makedirs(matched_folder)
verify_folder = 'matched_to_verify'
if not os.path.exists(verify_folder):
    os.makedirs(verify_folder)    


#read each image from data, get embedding for each face and comparing it to known face to check if the person we looking for is present or not
for filename in os.listdir(folder_path):
    if filename.endswith(".mp4") or filename.endswith(".MP4") :
        continue
    else:
        img_path = os.path.join(folder_path, filename)
        embeddings, boxes = get_embeddings(img_path)
        print(filename)
       
        if embeddings is not None:
            img_orignal = cv.imread(img_path)
            img=img_orignal.copy()
            verify_img=img_orignal.copy()
            l=0
            similarities = cosine_similarity( known_embeddings,embeddings)
            max_sim=np.max(similarities,axis=0)
            print(similarities.shape)
            print(len(boxes))
            with open('my_file.txt', 'w') as f:
                for item in similarities:
                    f.write("%s\n" % item)
            index=np.argmax(similarities,axis=0)
            p=np.where(max_sim>0.7)            
            id=[]
            for val in p[0]:

                l=l+1
                x1, y1, width, height = map(int, boxes[val])
                cv.rectangle(img, (x1, y1), (width, height), (0, 255, 0), 3)
                similarity_value = float(max_sim[val]) * 100
                similarity_value = round(similarity_value, 2)
                id.append(val)
                cv.putText(img, str(name[index[val]]), (x1, y1 - 100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7) 

            # if boxes is not None: 
            #     for f in range(len(boxes)):
            #             if f not in id:
            #                 x1, y1, width, height = map(int, boxes[f])
            #                 cv.rectangle(img, (x1, y1), (width, height), (0, 0, 255), 3)
                
            if l>0:
                cv.imwrite(str(matched_folder)+"/"+str(filename),img)
                x1=0

            verify=[i for i,z in enumerate(max_sim) if 0.67<z<0.7]
            if len(verify)>0:
                for v in verify:
                    x1, y1, width, height = map(int, boxes[v])
                    cv.rectangle(verify_img, (x1, y1), (width, height), (0, 255, 0), 3)  
                    similarity_value = float(max_sim[v]) * 100
                    similarity_value = round(similarity_value, 2)
                    cv.putText(verify_img, str(name[index[v]])+" :" +str(similarity_value)+ "%", (x1, y1 - 100), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7) 

                cv.imwrite(str(verify_folder)+"/"+str(filename),verify_img)
                      

end=timer()
ttime=end-start   
print(ttime)             
