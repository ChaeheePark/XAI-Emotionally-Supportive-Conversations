# Emotional-Support-Conversation

https://github.com/thu-coai/Emotional-Support-Conversation

### *Copyright © 2021 CoAI Group, Tsinghua University. All rights reserved. Data and codes are for academic research use only.*

Data and codes for the ACL 2021 paper: [**Towards Emotional Support Dialog Systems**](https://arxiv.org/abs/2106.01144)

If you use our codes or your research is related to our paper, please kindly cite our paper:

```bib
@inproceedings{liu-etal-2021-towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang  and 
    Zheng, Chujie  and 
    Demasi, Orianna  and 
    Sabour, Sahand  and 
    Li, Yu  and 
    Yu, Zhou  and 
    Jiang, Yong  and 
    Huang, Minlie},
  booktitle={Proceedings of the 59th annual meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Data

The corpus file is `ESConv.json`. We have collected **more** conversations with more problem topics. ESConv now contians 1,300 conversations with 10 topic problems.

`sentiment_chatbot_dataset_csv` : Add an emotion tag for each text.

### Statistics
#### Problem Category

| Problem Category | ongoing depression | breakup with partner | job crisis | problems with friends | academic pressure | procras-<br>tination* | alcohol abuse* | issues with parent* | sleep problems* |  appearance anxiety* | school bullying* | issues with children* |
| :-------- | :---------- | :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- | :---------- | :---------- | 
| Number| 351 | 239 | 280 | 179 | 156 |  13 | 12 | 18 | 28 | 12 | 2 | 10 |

#### Emotion Category
anger, anxiety, depression, disgust, fear, guilt, jealousy, nervousness, pain, sadness, shame, neutral

\* denotes the new topics added during the second collection. We hope new data supports the future research in transferring the ability of models from old topics to new ones. 

<font size=1>

#### Strategy Category 
| Strategy Category| Number   |
| :--------------  | :------- |
| Questions | 3801(20.7%)|
| Self-disclosure | 1713(9.3%) |
| Affirmation and Reassurance | 2827(15.4%) |
| Providing Suggestions | 2954(16.1%) |
| Other | 3661(18.3%) / 3341(18.2%) |
| Reflection of feelings |  1436(7.8%) | 
| Information | 1215(6.6%) | 
| Restatement or Paraphrasing | 1089(5.9%) |

</font>

## Preparing Enviroment

```bash
conda env create -f env.yml -n cuda
conda activate cuda
```

## Downloading Model

You should first download the [BlenderBot-small](https://huggingface.co/facebook/blenderbot_small-90M) model and replace the fake `pytorch_model.bin` file in `Blenderbot_small-90M` with the true one.

If you would like to evaluate generated results with Embedding-based similarity, you can download my prepared embedding files from [here](https://1drv.ms/f/s!Aky8v8NZbQx1qj7OlJKcQEJ6qrWm).

## Preprocessing Training Data

Run `bash RUN/prepare_strat_emotion.sh` to preprocess the training data.

## Training Your Model

Run `bash RUN/train_strat_emotion.sh` to train your model.

## Inference with Your Model

Every time of model training will create a new folder in `DATA/{inputter_name}.{config_name}`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/infer_strat_emotion.sh --load_checkpoint` with the path of your selected checkpoint.

Then, run `bash RUN/infer_strat_emotion.sh` to do the inference.

**Note**: When you run `infer_strat_emotion.sh`, you can change `GOLDEN_TRUTH` in  `inputters/PARAMS.py` to control whether use the golden strategy during inference.

## Interacting with Your Model

Similar to inference, after designating the checkpoint in `RUN/interact_strat_emotion.sh- -load_checkpoint`, run `bash RUN/interact_strat_emotion.sh`.
