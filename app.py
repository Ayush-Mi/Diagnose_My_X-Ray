import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from main import *
import matplotlib 
matplotlib.pyplot.switch_backend('Agg') 

app = Flask(__name__)
predmodel_path = './model'
image_path = './data/images/00000003_007.png'
last_layer_name = 'conv5_block3_3_bn'
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
       'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
       'Pneumothorax']
viz = Visualize()


model = tf.keras.models.load_model('./model/',
                                custom_objects={'weighted_loss':get_weighted_loss,
                                                'f1_m':f1_m,
                                                'precision_m': precision_m,
                                                'recall_m': recall_m})

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST': 
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(target_img , file.filename))
            img_path = os.path.join(target_img , file.filename)
            img = file.filename

            pre_img = load_image(img_path)
            indx, cls, prob = get_predictions(model,pre_img,labels,threshold=0)
            print(prob)
            sorted_prob = np.argsort(prob)[-3:]
            print(sorted_prob)

            pred_files = []
            for i in  sorted_prob:#range(len(cls)):
                x = viz.get_heat_map(model,pre_img, last_layer_name, pred_index=indx[i])
                save_img(x,title=cls[i]+'_'+str(prob[i]*100)[:5]+'%',file_name='./static/predictions/pred_'+img[:-4]+str(prob[i]*100)[:2]+'.png')
                pred_files.append('pred_'+img[:-4]+str(prob[i]*100)[:2]+'.png')


            predictions = {
                    "class1":cls[sorted_prob[2]],
                    "class2":cls[sorted_prob[1]],
                    "class3":cls[sorted_prob[0]],
                    "prob1": round(prob[sorted_prob[2]]*100,2),
                    "prob2": round(prob[sorted_prob[1]]*100,2),
                    "prob3": round(prob[sorted_prob[0]]*100,2),
            }

        else:
            error = "Please upload images of jpg , jpeg and png extension only"

        if(len(error) == 0):
            return  render_template('success.html' , img  = img , img_1 = pred_files[2], img_2=pred_files[1], img_3=pred_files[0], predictions = predictions)
        else:
            return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)


