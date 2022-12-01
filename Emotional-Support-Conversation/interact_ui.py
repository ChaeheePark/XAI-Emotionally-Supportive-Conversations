# coding=utf-8

import json
import datetime
import torch
from torch import Tensor
import numpy as np
import os
import logging
import argparse
import random
import csv
from transformers.trainer_utils import set_seed
from utils.building_utils import boolean_string, build_model, deploy_model
from inputters import inputters
from inputters.inputter_utils import _norm
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='strat_emotion')
parser.add_argument('--inputter_name', type=str, default='strat_emotion')
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)
parser.add_argument("--fp16", type=boolean_string, default=False)

parser.add_argument("--single_turn", action='store_true')
parser.add_argument("--max_input_length", type=int, default=256)
parser.add_argument("--max_src_turn", type=int, default=20)
parser.add_argument("--max_decoder_input_length", type=int, default=64)
parser.add_argument("--max_knl_len", type=int, default=64)
parser.add_argument('--label_num', type=int, default=None)

parser.add_argument("--min_length", type=int, default=10)
parser.add_argument("--max_length", type=int, default=40)

parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

parser.add_argument("--use_gpu", action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

if args.load_checkpoint is not None:
    output_dir = args.load_checkpoint + '_interact_dialogs'
else:
    os.makedirs('./DEMO', exist_ok=True)
    output_dir = './DEMO/' + args.config_name
    if args.single_turn:
        output_dir = output_dir + '_1turn'
os.makedirs(output_dir, exist_ok=True)

# set_seed(args.seed)

file_name = os.path.join('./history.csv')
fieldnames = ["Date", "Time", 'Speaker', "Text", "Emotion",
              "anger", "anxiety", "depression", "disgust", "fear", "guilt",
              "jealousy", "nervousness", "pain", "sadness", "shame", "neutral",
              "Strategy", "Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
              "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others"]
if not os.path.isfile(file_name):
    with open(file_name, 'a', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(["Date", "Time", 'Speaker', "Text", "Emotion",
                     "anger", "anxiety", "depression", "disgust", "fear", "guilt",
                     "jealousy", "nervousness", "pain", "sadness", "shame", "neutral",
                     "Strategy", "Question", "Restatement or Paraphrasing", "Reflection of feelings", "Self-disclosure",
                     "Affirmation and Reassurance", "Providing Suggestions", "Information", "Others"])
# set_seed(args.seed)

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

id2emotion = {0: 'anger', 1: 'anxiety', 2: 'depression', 3: 'disgust', 4: 'fear', 5: 'guilt',
              6: 'jealousy', 7: 'nervousness', 8: 'pain', 9: 'sadness', 10: 'shame', 11: 'neutral'}
id2strategy = {0: "Question", 1: "Restatement or Paraphrasing", 2: "Reflection of feelings", 3: "Self-disclosure",
               4: "Affirmation and Reassurance", 5: "Providing Suggestions", 6: "Information", 7: "Others"}

toker, model, *_ = build_model(checkpoint=args.load_checkpoint, **names)
model = deploy_model(model, args)

model.eval()

inputter = inputters[args.inputter_name]()
dataloader_kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knl_len': args.max_knl_len,
    'label_num': args.label_num,
}


pad = toker.pad_token_id
if pad is None:
    pad = toker.eos_token_id
    assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
bos = toker.bos_token_id
if bos is None:
    bos = toker.cls_token_id
    assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
eos = toker.eos_token_id
if eos is None:
    eos = toker.sep_token_id
    assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}

today = datetime.date.today()
today = today.isoformat()
history = {'dialog': [], }

main_ui = 'UI/chat.UI'
form_1, base_1 = uic.loadUiType(main_ui)


class main_page(base_1, form_1):
    def __init__(self):
        super(base_1, self).__init__()
        self.setupUi(self)

        self.cnt_in = 0
        self.cnt_out = 0
        self.check = False
        self.humans = [self.human1, self.human2, self.human3]
        self.emotion = [self.emotion1, self.emotion2, self.emotion3]
        self.ais = [self.ai1, self.ai2, self.ai3]
        self.strategy = [self.strategy1, self.strategy2, self.strategy3]

        self.show()
        self.enter_bt.clicked.connect(self.button_event)

    def button_event(self):
        cur_t = datetime.datetime.now()
        cur_t_in = str(cur_t.hour) + "h " + str(cur_t.minute) + "m"
        in_text = self.input_text.toPlainText()
        h_texts = [self.humans[1].toPlainText(), self.humans[2].toPlainText()]
        emos = [self.emotion[1].text(), self.emotion[2].text()]
        a_texts = [self.ais[1].toPlainText(), self.ais[2].toPlainText()]
        strats = [self.strategy[1].text(), self.strategy[2].text()]

        if not self.check:
            self.humans[self.cnt_in].setText(in_text)
            if self.cnt_in == 2:
                self.check = True
            else:
                self.cnt_in += 1
        else:
            self.humans[0].setText(h_texts[0])
            self.emotion[0].setText(emos[0])
            self.ais[0].setText(a_texts[0])
            self.strategy[0].setText(strats[0])
            self.humans[1].setText(h_texts[1])
            self.emotion[1].setText(emos[1])
            self.ais[1].setText(a_texts[1])
            self.strategy[1].setText(strats[1])
            self.humans[2].setText(in_text)
            self.emotion[2].clear()
            self.ais[2].clear()
            self.strategy[2].clear()
        self.input_text.clear()

        history['dialog'].append({
            'text': _norm(in_text),
            'speaker': 'usr'
        })
        with open(file_name, 'a', newline='') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow([today, cur_t_in, 'usr', _norm(in_text)])

        # generate response
        history['dialog'].append({  # dummy tgt
            'text': 'n/a',
            'speaker': 'sys',
            'emotion': 'neutral',
            "strategy": "Others"
        })
        inputs = inputter.convert_data_to_inputs(history, toker, **dataloader_kwargs)
        inputs = inputs[-1:]
        features = inputter.convert_inputs_to_features(inputs, toker, **dataloader_kwargs)
        batch = inputter.prepare_infer_batch(features, toker, interact=True)
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update(generation_kwargs)
        encoded_info, generations = model.generate(**batch)

        out = generations[0].tolist()
        out = cut_seq_to_eos(out, eos)
        out_text = toker.decode(out).encode('ascii', 'ignore').decode('ascii').strip()
        cur_t = datetime.datetime.now()
        cur_t_out = str(cur_t.hour) + "h " + str(cur_t.minute) + "m"

        emotion_id_out = encoded_info['pred_emotion_id_top3'].tolist()[0][0]
        emotion_dist = encoded_info['pred_emotion_id_dist'].numpy()[0]
        e = np.round(emotion_dist, 3)
        emotion = id2emotion[emotion_id_out]

        strat_id_out = encoded_info['pred_strat_id_top3'].tolist()[0][0]
        strat_dist = encoded_info['pred_strat_id_dist'].numpy()[0]
        s = np.round(strat_dist, 3)
        strategy = id2strategy[strat_id_out]

        if self.cnt_out != 2:
            self.ais[self.cnt_out].setText(out_text)
            self.emotion[self.cnt_out].setText(emotion)
            self.strategy[self.cnt_out].setText(strategy)
            self.cnt_out += 1
        else:
            self.ais[2].setText(out_text)
            self.emotion[2].setText(emotion)
            self.strategy[2].setText(strategy)

        with open(file_name, 'a', newline='') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow([today, cur_t_out, 'sys', out_text, emotion,
                         e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11],
                         strategy, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])

        history['dialog'].pop()
        history['dialog'].append({
            'text': out_text,
            'speaker': 'sys',
            'emotion': emotion,
            "strategy": strategy
        })


if __name__ == '__main__':
    app = QApplication(sys.argv)
    op = main_page()
    op.show()
    app.exec_()
