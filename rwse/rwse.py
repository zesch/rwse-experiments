from cassis import Cas
from cassis.typesystem import TypeNotFoundError
from transformers import pipeline
from typing import List
import csv

T_SENTENCE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
T_RWSE = 'de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE'
T_TOKEN = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'

class RWSE_Checker:

    def __init__(self) -> None:
        # TODO make language aware
        self.pipe = pipeline("fill-mask", model="bert-base-cased")

    def _load_confusion_sets(self, file_path):
        confusion_sets = {}
        with open(file_path, 'r') as file:
            for row in csv.reader(file):
                for key in row:
                    confusion_sets[key] = list(row)
        return confusion_sets
    
    def _create_masked_sentence(self, cas, i, tokens):
        text = cas.sofa_string

        token = tokens[i]

        prefix_start = tokens[0].begin
        prefix_end = tokens[0].begin
        if i > 0:
            prefix_end = tokens[i-1].end

        suffix_start = tokens[len(tokens)-1].end
        suffix_end = tokens[len(tokens)-1].end
        if i < len(tokens)-1:
            suffix_start = tokens[i+1].begin

        # get string from start to token and after token to end
        prefix = text[prefix_start:prefix_end]
        suffix = text[suffix_start:suffix_end]

        return prefix + " [MASK] " + suffix

    def set_confusion_sets(self, filepath):
        self.confusion_sets = self._load_confusion_sets(filepath)

    # also called automatically within check, but might be useful to have exposed,
    # e.g. when wanting to show which tokens are in confusion sets without running the (costly) pipeline
    def needs_checking(self, token):
        return token in self.confusion_sets 
    
    def check_cas(self, cas: Cas, ts):
        try:
            RWSE = ts.get_type(T_RWSE)
        except TypeNotFoundError:
            print("RWSE type not found")
            ts.create_type(T_RWSE)
            RWSE = ts.get_type(T_RWSE)

        for sentence in cas.select(T_SENTENCE):
            tokens = [token for token in cas.select_covered(T_TOKEN, sentence)]
            for i in range(len(tokens)):
                masked_sentence = self._create_masked_sentence(cas, i, tokens)

                token_str = tokens[i].get_covered_text()
                correct_token_str = self.check(token_str, masked_sentence=masked_sentence)
                if token_str != correct_token_str:
                    # TODO set correction feature
                    cas_rwse = RWSE(begin=tokens[i].begin, end=tokens[i].end)
                    cas.add(cas_rwse)
                    print(cas_rwse)
        #return cas

    def check(self, token: str, masked_sentence: str) -> str:
        # if no need to check return token
        if not self.needs_checking(token):
            return token
        
        # In order to be sure about a change, we should be orders of magnitude more probable
        magnitude = 10^1
        correct_token = token
        highest_prob = 0.0
        target_prob = 0.0
        results = self.pipe(masked_sentence, targets=self.confusion_sets[token])
        for result in results:
            if result["token_str"] == token:
                target_prob = min(result["score"] * magnitude, 1.0)
        for result in results:
            # print(f'{result["token_str"]}: {result["score"]}')
            if result["score"] > target_prob and result["score"] > highest_prob:
                highest_prob = result["score"]
                correct_token = result["token_str"]

        return correct_token

if __name__ == "__main__":
    rwse = RWSE_Checker()
    rwse.set_confusion_sets('../data/confusion_sets.csv')
    token = "there"
    masked_sentence = "I want to buy [MASK] cars."
    correct_token = rwse.check(token, masked_sentence)
    print(token, correct_token)