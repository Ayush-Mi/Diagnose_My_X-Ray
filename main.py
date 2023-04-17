import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
from visualize import Visualize


## Custom layers used in the model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-3):

    def weighted_loss(y_true, y_pred):
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += - pos_weights[i] * K.mean(y_true[:,i] * K.log(y_pred[:,i] + epsilon)) \
            - neg_weights[i] * K.mean((1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon))
        return loss
    
    return weighted_loss

## Helper functions
def load_image(path):
    image = tf.keras.preprocessing.image.load_img(path,target_size=(320,320))
    return np.expand_dims(image,axis=0)

def get_predictions(model,image,labels,threshold=0.6):
    prediction = model.predict(image)

    pred_index = (np.where(prediction[0]>threshold))[0]
    pred_probablity = prediction[prediction>=threshold]
    pred_class = [labels[x] for x in pred_index]

    return pred_index, pred_class, pred_probablity

def save_img(img,title,file_name):
    plt.figure(figsize=(10, 10))
    plt.title(title,fontsize=20)
    plt.imshow(img)
    plt.savefig(file_name)
    print('Prediction saved at ',file_name)
    


## Variable declaration
model_path = './model'
image_path = './data/images/00000003_007.png'
last_layer_name = 'conv5_block3_3_bn'
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
       'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
       'Pneumothorax']
viz = Visualize()


if __name__ == '__main__':
    model = tf.keras.models.load_model('./model/',
                                   custom_objects={'weighted_loss':get_weighted_loss,
                                                    'f1_m':f1_m,
                                                    'precision_m': precision_m,
                                                    'recall_m': recall_m})
    pre_img = load_image(image_path)

    indx, cls, prob = get_predictions(model,pre_img,labels)
    
    for i in range(len(cls)):
        x = viz.get_heat_map(model,pre_img, last_layer_name, pred_index=indx[i])
        save_img(x,title=cls[i]+'_'+str(prob[0]*100)[:5]+'%',file_name='./data/predictions/'+str(prob[i]*100)[:2]+'.png')

