import json

from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider

from album_eval import AlbumEvaluator

ref_json_path = './data/test_reference.json'
reference = json.load(open(ref_json_path))

predictions = {}
prediction_file = './data/prediction_test'
with open(prediction_file) as f:
    for line in f:
        vid, seq = line.strip().split('\t') # split into id and story
        if vid not in predictions:
            predictions[vid] = [seq]

# # test sentence #0.7398
# ref =  'It is a guide to action that ensures that the military will forever heed Party commands'
# hypo = 'It is a guide to action which ensures that the military always obeys the commands of the party'
# reference =   {"000": [ref]}
# predictions = {"000": [hypo]}

myeval = AlbumEvaluator()
myeval.evaluate(reference, predictions)