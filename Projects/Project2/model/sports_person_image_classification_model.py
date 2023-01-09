import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import pywt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
import joblib
import json

img = cv2.imread("./test_images/sharapova1.jpg")

plt.imshow(img)
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()

face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

(x,y,w,h)=faces[0]

face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
plt.imshow(face_img)
plt.show()

cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()

plt.imshow(roi_color, cmap='gray')
plt.show()

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) != 1):
        return None
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes) >= 2):
            return roi_color

cropped_image = get_cropped_image_if_2_eyes("./test_images/sharapova1.jpg")
plt.imshow(cropped_image)
plt.show()      

path_to_data = "./dataset/"
path_to_cr_data = "./dataset/cropped/"

image_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        image_dirs.append(entry.path)

if not os.path.exists(path_to_cr_data):
    os.makedirs(path_to_cr_data)
    for image_dir in image_dirs:
        count = 0
        for entry in os.scandir(image_dir):
            roi_color = get_cropped_image_if_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder_path = os.path.join(path_to_cr_data, image_dir.split("/")[-1])
                if not os.path.exists(cropped_folder_path):
                    os.makedirs(cropped_folder_path)
                cropped_file_path = os.path.join(cropped_folder_path, str(count)+".png")
                cv2.imwrite(cropped_file_path, roi_color)
                count += 1

cropped_image_dirs = []
for entry in os.scandir(path_to_cr_data):
    if entry.is_dir():
        cropped_image_dirs.append(entry.path)

celebrity_file_names_dict = {}
for image_dir in cropped_image_dirs:
    celebrity_name = image_dir.split("/")[-1]
    file_names = []
    for entry in os.scandir(image_dir):
        if entry.is_file():
            file_names.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_names

def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

im_har = w2d(cropped_image,'db1',5)
plt.imshow(im_har, cmap='gray')
plt.show()

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count += 1

X=[]
Y=[]

for celebrity_name,training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        im = cv2.imread(training_image)
        scaled_raw_img = cv2.resize(im, (32,32))
        im_har = w2d(im,'db1',5)
        scaled_img_har = cv2.resize(im_har, (32,32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        Y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X),4096).astype(float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])

pipe.fit(X_train, Y_train)

print(pipe.score(X_test, Y_test))

model_params = {
    'svm': {
        'model': SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,20],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params' : {
            'logisticregression__C': [1,5,10]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "kneighborsclassifier__n_neighbors": [1, 5, 10]
        }
    }
}

scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, Y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

best_clf = best_estimators['svm']

joblib.dump(best_clf, 'saved_model.pkl')

with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))






