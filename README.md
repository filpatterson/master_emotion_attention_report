# master_emotion_attention_report

Current section is dedicated for regression models that tried to predict 
```valence``` and ```arousal``` variables covering the emotional state in 
a continuous form and final classification model that used some of the concepts
tested on regression models.

**Important:** *models created in the result of classification models research and regression
model research are not located on the Git because of the size limitations. Because of that,
you are able to recreate those models on your own. In case of a need to verify those models,
author will provide them to the reviewer (from the university). Consider that demonstrated work
is a result of a master thesis work and it is not covering complete solution that can be
thrown to the market, there is a lot of work to do.*

There are 3 packages: ```models``` demonstrates base concept of how the models look like (there
are demonstrated the most primitive developed models), ```Trained Networks``` taken from the 
```Affect-Net``` dataset in case if reviewer would like to compare its efficiency against 
developed solution, and ```master_degree_bump``` covering final stage of creating the application.

There are separate sections of the final application for demonstration that will are described
in a short text below.

# Regression research

Datasets contained variables ```valence``` and ```arousal``` covering emotional state of the
person. Considering their continuous form, the perfect model will be trained over those
parameters and based on those parameters consider presented emotion. The problem is that
those parameters are subjective, there is no efficient formula (as I know) at the moment capable
of exact estimation of those continuous variables considering human face expression. Because of
their unclear estimation nature, research was performed in a short time just to consider this
option and check how efficient it will perform.

Achieved results were able to explain only 40-45% of the both ```valence``` and ```arousal```
variables distribution, meaning that model is not understanding how those variables correlate
with the face expression. Because of that, it was decided to reject in using regression model.
Reason of such result can be estimated because of constructing inefficient model, or because
variables were estimated with errors.

# Final classification model

The final emotion classification model is performing based on combining best characteristics
of researched classification and regression approaches. As input data were considered 3D
scans of the human faces made out of the original images using the MediaPipe FaceMesh model.
Using this approach author was able to achieve a 65% accuracy of the classification, meaning
that this solution can be considered for further creation of complex application making
emotion classification. It will require additional verifications, but it is possible to create
great solution. 

This classification model will be used/integrated in the final application.

# Final application

Final application is a Qt based Python application with integrated emotion classification
model and pipelines for preprocessing the data. Application contains video player (to play
content for which reaction is required), webcam stream to get face of the person, model
for extracting emotion and forming 3D face scan, and recorder of the reaction. Make sure
that you installed all required libraries to make this code work, that you are using
Python 3.9. Launch of the code can be done via ```main.py``` and entire app is located inside
of the ```final_app``` package inside of the ```master_degree_bump```