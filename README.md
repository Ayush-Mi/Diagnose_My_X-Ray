# Diagnose_My_X-Ray

It is estimated that nearly 100 million x-rays are performed every year in just India - source ['Times of India'](http://timesofindia.indiatimes.com/articleshow/4262527.cms?utm_source=contentofinterest&utm_medium=text&utm_campaign=cppst). With this rise in this number the analysis of these reports forms the bottleneck of the whole diagnostic process. In this porject I have tried to automate this analysis process to help  doctors in analysing these reports by highlighting the areas which has abnormility and classify them to the decease that they indicate along with the probability of that decease.

This does not eliminate the requirement of a clinician but instead aid them by highlighting parts of interest in the report and reduce the time it take for analysis.

#### Demo

This is a short demo of the web application where we can upload the image of our x-ray and it predicts the decease which the person might contain along with evidence which supports this prediction.
![](https://github.com/Ayush-Mi/Diagnose_My_X-Ray/blob/main/static/for_readme/final_diagnosis.gif)

#### Final Predictions

A screenshot of what the final prediction looks like.
![](https://github.com/Ayush-Mi/Diagnose_My_X-Ray/blob/main/static/for_readme/prediction.png)

## How it was made?

The whole project is divided into three parts: training a model, visualizing the output and creating a web based app where one can upload an x-ray image and get the analysis/predictions. It uses a deep learning model trained on labeled x-ray dataset to analyse a given image. This can further be integrated with a NLP model to auto-generate text for non-medical person to understand the reports.

#### Model
The last 15 layers of a Keras based ResNet50 model pretrained on Imagenet dataset was finetuned on the x-ray data. The model was trained on weighted loss function with Adam optimizer and F1-score as the evaluation metric for 10 epochs. It had 23.6 M parameters out of which 5.5 M were trainable parameters and each epoch took around 10 mins when trained on Macbook M1-pro 32gb.

#### Dataset
The image data used for training was a subset taken from the paper [ChestX-ray8](https://arxiv.org/abs/1705.02315) by Xiaosong Wang et.al. It had ~15k chest x-ray images of ~3.9k patients beloning to one or more of the 15 labels i.e. 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema','Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration','Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia','Pneumothorax'.

However, the frequency of occurance of these classes in the whole dataset was really low and hence a weighted loss function was used while training the model.

#### Results

The table shows the best results while training which was seen in epoch 8 of 10. The model clearly overfits the data in training pipeline which can be seen by the high difference in the train and val metrics.

| | Loss | F1 | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|
| Train | 0.26 | 0.67 | 0.52 | 0.94 |
| Val | 3.59 | 0.36 | 0.31 | 0.43 |
| Test | 2.46 | 0.40 | 0.36 | 0.46 |


## How to run this demo?
The latest versions of
## Room for improvement

- The image data can be preprocessed to improve the quality of the image which in turn will help the model learn better.
- The DICOM images are of high quality and hence are more suitable to get a better performing model.
- The above aprroach used the resnet50 as the baseline but more recent architectures like transformers can capture patterns well.

## References
- I ain't an expert in frontend so the CSS and HTML was taken from [BuffML](https://buffml.com/multi-class-image-classification-flask-app-complete-project/)
- The dataset used was taken from [ChestX-ray8](https://arxiv.org/abs/1705.02315)
