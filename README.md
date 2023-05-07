# On the Deep Generative Models Explaining the Rationale to Emotionally Supportive Conversations

by *Eunhye Jeong^, Chaehee Park^, Hyejin Hong^ and Jeehang Lee** (^ = equal contribution in the paper)

***Best Paper Award in KOSES Autumn Conference 2022, Busan***

This repository contains all the documented scripts and files (or links for downloading some resources) to run our experiments.

Since the growing number of people suffering from mental health problems has limited opportunity to have psychological treatment, chabots research has paid much attention as a ubiquitous means to exhibit effective psychological support. However, it is not clear how and why the chatbots determine such supportive content to provide, which significantly diminishes the trust and efficacy. Thus, we propose a novel deep generative model which delivers circumstantial information in the generation of the treatment content.

Emotional supportive chatbot creates emotional conversations in response to the user's input conversation. At this time, the rationale for the response generation of the chatbot is predicted and presented together with emotional conversation. The model primarily generates emotionally supportive dialogue in response to the user input, whilst it infers the type of user’s emotion, its intensity and the treatment strategy as a rationale. These three information are presented as explainbale information for chatbot conversations, forming a system that can explain the emotional supportive content of the chatbot.

The chatbot, along with dialogue, outputs emotion for the sentence by user input and the strategy used by the chatbot to provide emotional support to the user. For emotion classification on the user input, we included all sentences with ESConv, an Emotion Supportive Conversation dataset, labeled as one of 12 emotions for all texts based on the *emotion_type* provided by ESConv. In addition, in order to output the emotional intensity of the user input, the user input was used as the test data of the Neural Emotion Intensity Prediction to output the emotional intensity of the four emotions (Anger, Sadness, Fear, Joy).

We trained and run each model of **Emotion Supportive Chatbot** and **Neural Emotion Intensity**. It is shown below how to run each model.

*Details can be found in our paper (with the title above) accepted for publication at KOSES Autumn Confenrence 2022. The PDF is available [here](https://drive.google.com/file/d/15Q02Gsxfv0eDoLHcsyffxQ0fhC9klhR-/view?usp=sharing).*
#

## Emotional Support Conversation
### To successfully and smoothly run our experiments, please follow the steps below - 

1. Preparing Enviroment
```bash
conda env create -f env.yml -n cuda
conda activate cuda
```
2. You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and replace the fake `pytorch_model.bin` file in `Blenderbot_small-90M` with the true one. If you would like to evaluate generated results with Embedding-based similarity, you can download my prepared embedding files from [here](https://1drv.ms/f/s!Aky8v8NZbQx1qj7OlJKcQEJ6qrWm).
3. Run `bash RUN/prepare_strat_emotion.sh` to preprocess the training data.
4. Train / inference / interact
- Train: Run `bash RUN/train_strat_emotion.sh` to train your model.
- inference: Every time of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/infer_strat_emotion.sh --load_checkpoint` with the path of your selected checkpoint. Then, run `bash RUN/infer_strat_emotion.sh` to do the inference.
- interact: Similar to inference, after designating the checkpoint in `RUN/interact_strat_emotion.sh- -load_checkpoint`, run `bash RUN/interact_strat_emotion.sh`.



## Neural Emotion Intensity Prediction
### To successfully and smoothly run our experiments, please follow the steps below - 

1. Collect all the dialogues of user inputs and Head to *XAI-Emotionally-Supportive-Conversations/Neural-Emotion-Intensity-Prediction/data/test* directory.
2. You will see a total of 4 files in *eval(anger, pear, joy, sadness).txt* in the current directory.
3. If you open a file(ex: *evalanger.txt*), you can see that there are sentences in the second column, and replace them with all the dialogues of user inputs. Note that you need to replace the user inputs from the top, and the rest should be the original set length. (Reference Code: ~) Repeat for all 4 files.
4. Head to *XAI-Emotionally-Supportive Conversations/Neural-Emotion-Intensity-Prediction* and follow the instructions in that README to number 5.
5. Run *codes/Multi_task/LE_PC_DMTL_EI_Demo.py* to see the intensity of each of anger, pear, joy, and sadness. At this time, `y_pred_(anger, pear, joy, sadness)` at the end of the code can be checked by using `print` as many as the number of user inputs.(ex:`print(y_pred_joy[:10])` )
