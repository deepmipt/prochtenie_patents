import itertools
import json
import re

import pymorphy2
from deeppavlov.core.data.utils import simple_download


def morph_parse(word):
    parsed_tok = morph.parse(word)[0]
    normal_form = parsed_tok.normal_form
    return normal_form


morph = pymorphy2.MorphAnalyzer()
tokenize_reg = re.compile(r"[\w']+|[^\w ]")

url = "http://files.deeppavlov.ai/prochtenie/cause_effect.json"
database_flname = "/model_data/cause_effect.json"
simple_download(url, database_flname)
cause_effects = json.load(open(database_flname, "r", encoding="utf-8"))


def check_cause_effect(paragraphs, topic):
    years = re.findall("[\d]{3,4}", topic)
    k3_points = 2
    if len(years) == 2:
        period = f"{years[0]}-{years[1]}"
        if period in cause_effects:
            k3_points = 0
            for par in paragraphs:
                essay_tokens = list(itertools.chain.from_iterable([elem["words"] for elem in par]))
                essay_tokens = [morph_parse(tok).lower() for tok in essay_tokens]
                par = " ".join(essay_tokens)
                causes_and_effects = cause_effects[period]
                for causes, effects in causes_and_effects:
                    found_cause = False
                    found_effect = False
                    for cause in causes:
                        cause = " ".join([morph_parse(tok).lower() for tok in cause.split()])
                        if cause in par:
                            found_cause = True

                    for effect in effects:
                        effect = " ".join([morph_parse(tok).lower() for tok in effect.split()])
                        if effect in par:
                            found_effect = True

                    if found_cause and found_effect:
                        k3_points += 1
                        break
    return k3_points
