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

face_folder_path = "your folder location where you stored images with single face and given name of person as file name for label"


#intializing model and MTCNN for face recognition
resnet = InceptionResnetV1(pretrained='vggface2',device=device).eval()
mtcnn = MTCNN(device=device)


#getting embbedings of the faces we need to remember
known_embeddings = []
name=[]
for filename in os.listdir(face_folder_path):
    if filename.endswith(".mp4"):
        continue
    else:
                person1_name = os.path.splitext(os.path.basename(filename))[0].split('_')[0]
                name.append(person1_name)
                img_path = os.path.join(face_folder_path, filename)
                img = cv.imread(img_path)
                img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                # from PIL import Image, ExifTags

                # image = Image.open(img_path).convert('RGB')

                # # Check if the image has an EXIF orientation tag
                # if hasattr(image, '_getexif'):
                #     exif = image._getexif()
                #     if exif is not None:
                #         for orientation in ExifTags.TAGS.keys():
                #             if ExifTags.TAGS[orientation] == 'Orientation':
                #                 if orientation in exif:
                #                     if exif[orientation] == 3:
                #                         image = image.rotate(180, expand=True)
                #                     elif exif[orientation] == 6:
                #                         image = image.rotate(270, expand=True)
                #                     elif exif[orientation] == 8:
                #                         image = image.rotate(90, expand=True)
                #                         print("inside")
                #                     image = image.transpose(Image.FLIP_LEFT_RIGHT)
                #                     break

                # # # Save the image
                # # image.save('path/to/image.jpg')
                # img1=image
                faces = mtcnn(img1)
                print("Processing image " + filename)

                if faces is None:
                    print("No faces found in "+filename)
                    continue
                num_faces = len(faces)    
                print("Found "+str(num_faces)+ " faces in" + str(filename))        
                emb = resnet(faces.unsqueeze(0)).detach().numpy()[0]
                known_embeddings.append(emb)


data = (known_embeddings, name)
with open('my_data.pkl', 'wb') as f:
    pickle.dump(data, f)