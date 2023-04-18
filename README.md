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

#### Training

#### Result

## How to run this demo?

## References
- I ain't an expert in frontend so the CSS and HTML was taken from [BuffML](https://buffml.com/multi-class-image-classification-flask-app-complete-project/)
- The dataset used was taken from [ChestX-ray8](https://arxiv.org/abs/1705.02315)
