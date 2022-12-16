# On the Deep Generative Models Explaining the Rationale to Emotionally Supportive Conversations

by *Eunhye Jeong^, Chaehee Park^, Hyejin Hong and Jeehang Lee** (^ = equal contribution in the paper)

<**Best Paper Award in KOSES 2022 Korea, Busan**>

This repository contains all the documented scripts and files (or links for downloading some resources) to run our experiments.

Since the growing number of people suffering from mental health problems has limited opportunity to have psychological treatment, chabots research has paid much attention as a ubiquitous means to exhibit effective psychological support. However, it is not clear how and why the chatbots determine such supportive content to provide, which significantly diminishes the trust and efficacy. Thus, we propose a novel deep generative model which delivers circumstantial information in the generation of the treatment content.

Emotional supportive chatbot creates emotional conversations in response to the user's input conversation. At this time, the rationale for the response generation of the chatbot is predicted and presented together with emotional conversation. The model primarily generates emotionally supportive dialogue in response to the user input, whilst it infers the type of user’s emotion, its intensity and the treatment strategy as a rationale. These three information are presented as explainbale information for chatbot conversations, forming a system that can explain the emotional supportive content of the chatbot.

The chatbot, along with dialogue, outputs emotion for the sentence by user input and the strategy used by the chatbot to provide emotional support to the user. For emotion classification on the user input, we included all sentences with ESConv, an Emotion Supportive Conversation dataset, labeled as one of 12 emotions for all texts based on the *emotion_type* provided by ESConv.
In addition, in order to output the emotional intensity of the user input, the user input was used as the test data of the Neutral Emotion Intensity Prediction to output the emotional intensity of the four emotions (Anger, Sadness, Fear, Joy).

We trained and run each model of **Emotion Supportive Chatbot** and **Neutral Emotion Intensity**. It is shown below how to run each model.

*Details can be found in our paper (with the title above) accepted for publication at koses 2022. The PDF is available [here](https://drive.google.com/file/d/15Q02Gsxfv0eDoLHcsyffxQ0fhC9klhR-/view?usp=sharing).*
#

## Emotional Support Conversation
### To successfully and smoothly run our experiments, please follow the steps below - 

1. 여기에 Emotional support Conversation Run 하는 방법 적어주세용
2. 🖤


## Neutral Emotion Intensity Prediction
### To successfully and smoothly run our experiments, please follow the steps below - 

1. Intensity 하는 방법 적어주세용
2. 🖤
