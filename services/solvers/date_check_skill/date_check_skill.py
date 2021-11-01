# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from io import StringIO

import nltk
import numpy as np
import pymorphy2
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from udapi.block.read.conllu import Conllu


@register("event_matcher")
class EventMatcher:
    def __init__(self, history_dates_file: str, *args, **kwargs):
        self.history_dates_file = history_dates_file
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя- 1234567890XIV–"

        with open(str(expand_path(self.history_dates_file)), "r") as fl:
            dates_base = json.load(fl)

        self.events = [entry["event"] for entry in dates_base]
        self.dates = [entry["dates"] for entry in dates_base]
        events_prepr = [self.text_to_normal_form(event) for event in self.events]

        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=100000)
        self.vectorizer.fit(events_prepr)
        self.vectorizer.get_feature_names()
        self.matrix = self.vectorizer.transform(events_prepr).transpose()
        self.filtered_event_nums = list(range(len(self.events)))

    def filter(self, start_year, end_year):
        filtered_event_nums = []
        for n, date_list in enumerate(self.dates):
            for date in date_list:
                if "date" in date.keys():
                    year = date["date"]["year"]
                    if year >= start_year and year <= end_year:
                        filtered_event_nums.append(n)
                if "date_interval" in date.keys():
                    if "year" in date["date_interval"]["start"]:
                        cand_year_start = date["date_interval"]["start"]["year"]
                        cand_year_end = date["date_interval"]["end"]["year"]
                    if "value" in date["date_interval"]["start"]:
                        cand_year_start = date["date_interval"]["start"]["value"]
                        cand_year_end = date["date_interval"]["end"]["value"]
                    if (cand_year_start >= start_year and cand_year_start <= end_year) or (
                        cand_year_end >= start_year and cand_year_end <= end_year
                    ):
                        filtered_event_nums.append(n)
        self.filtered_event_nums = sorted(list(set(filtered_event_nums)))

    def search(self, query):
        thres = 100
        query = self.text_to_normal_form(query)
        q_matrix = self.vectorizer.transform([query])
        scores = q_matrix * self.matrix
        scores = np.squeeze(scores.toarray() + 0.0001)
        o = np.argpartition(-scores, thres)[0:thres]
        o_sort = o[np.argsort(-scores[o])]
        candidates = [self.events[i] for i in o_sort if i in self.filtered_event_nums]
        dates = [self.dates[i] for i in o_sort if i in self.filtered_event_nums]
        scores = sorted(scores, reverse=True)
        return candidates, dates, scores

    def morph_parse(self, word):
        parsed_tok = self.morph.parse(word)[0]
        normal_form = parsed_tok.normal_form
        return normal_form

    def text_to_normal_form(self, text):
        text_sanitized = ""
        for letter in text:
            if letter in self.alphabet:
                text_sanitized += letter
        text_sanitized = text.replace("  ", " ")
        text_sanitized = " ".join(list(set([self.morph_parse(word) for word in text_sanitized.split(" ")])))
        return text_sanitized


@register("date_checker")
class DateChecker:
    def __init__(self, history_dates_file: str, batch_size: int = 30, tfidf_dot_product_thres: float = 0.2, **kwargs):
        self.batch_size = batch_size
        self.history_dates_file = history_dates_file
        self.morph = pymorphy2.MorphAnalyzer()
        self.event_matcher = EventMatcher(history_dates_file=self.history_dates_file)
        nltk.download("stopwords")
        self.stopwords = set(nltk.corpus.stopwords.words("russian"))
        self.tfidf_dot_product_thres = tfidf_dot_product_thres

    def __call__(self, sentences_init, parses, topic, offsets):
        period = re.findall("[\d]{3,4}", topic)
        if period and len(period) == 2:
            period_start, period_end = period
            self.event_matcher.filter(period_start, period_end)
        match = False
        errors = []
        dates = [self.find_date(sentence) for sentence in sentences_init]
        sentences = []
        for sentence, date_list in zip(sentences_init, dates):
            for date in date_list:
                sentence = sentence.replace(f"({date[1]})", f"{date[1]}")
            sentence = self.sanitize_punctuation(sentence)
            sentences.append(sentence)

        if len(parses) > 0 and isinstance(parses[0], list):
            parses = ["\n".join("\t".join(elem) for elem in parse) for parse in parses]
        if len(parses) > 0 and isinstance(parses[0], str):
            trees = [Conllu(filehandle=StringIO(parse)).read_tree() for parse in parses]

        for sentence, offset, tree, date_list in zip(sentences, offsets, trees, dates):
            for date in date_list:
                match = self.check_date(tree, date)
                if not match:
                    error = {}
                    error["type"] = "И.факт"
                    date_str = date[1]
                    start_span = offset[0] + sentence.find(date_str)
                    end_span = start_span + len(date_str)
                    error["text"] = date_str
                    error["start_span"] = start_span
                    error["end_span"] = end_span
                    errors.append(error)

        return errors

    def sanitize_punctuation(self, text):
        text = text.replace('=""', "")
        patterns = []
        for i in range(len(text) - 3):
            if text[i].islower() and text[(i + 1)] == "," and text[(i + 2)].islower():
                patterns.append((text[i : i + 3], f"{text[i:i+2]} {text[i+2]}"))
            if text[i].islower() and text[(i + 1)] == " " and text[(i + 2)] == ",":
                patterns.append((text[i : i + 3], f"{text[i]}{text[i+2]}"))
            if text[i].islower() and text[(i + 1)] == "." and text[(i + 2)].isupper():
                patterns.append((text[i : i + 3], f"{text[i:i+2]} {text[i+2]}"))
            if text[i].islower() and text[i + 1 : i + 3] == " ." and text[i + 3].isupper():
                patterns.append((text[i : i + 4], f"{text[i]}. {text[i+3]}"))
        for pattern in patterns:
            text = text.replace(pattern[0], pattern[1])
        return text

    def link_event(self, tree, date):
        date_info, date_str = date
        candidate_events = []
        candidate_dates = []
        scores = []
        if date_info:
            tree_tokens = []
            event_tokens = []

            for node in tree.descendants:
                tree_tokens.append(node.form)
            date_str_tokens = date_str.split(" ")

            date_str_start = 0
            for i in range(len(tree_tokens) - len(date_str_tokens)):
                count = 0
                for j in range(len(date_str_tokens)):
                    if fuzz.ratio(tree_tokens[(i + j)], date_str_tokens[j]) >= 80.0:
                        count += 1
                if count == len(date_str_tokens):
                    date_str_start = i + 1
                    for k in range(len(date_str_tokens)):
                        event_tokens.append((tree_tokens[(i + k)], i + k + 1))
                    break

            date_start_node = ""

            for node in tree.descendants:
                if node.ord == date_str_start:
                    date_start_node = node

            if date_start_node:
                parent_node = date_start_node
                parent_node_prev = date_start_node
                while parent_node.deprel not in ["root", "parataxis", "conj", "nsubj", "<ROOT>"]:
                    if parent_node.deprel == "acl":
                        parent_node_prev = parent_node
                        parent_node = parent_node.parent
                        break
                    parent_node_prev = parent_node
                    parent_node = parent_node.parent

                event_tokens.append((parent_node.form, parent_node.ord))

                found_subj = False
                for node in parent_node.children:
                    if node.deprel == "nsubj":
                        found_subj = True
                        event_tokens.append((node.form, node.ord))
                        event_tokens = self.make_event_tokens_list(node, event_tokens)

                event_tokens = self.make_event_tokens_list(parent_node_prev, event_tokens)

                is_a_found = False
                is_a_cnt = 0
                for node in parent_node.children:
                    if node.form == "это":
                        is_a_cnt += 1
                    if node.deprel == "nummod":
                        is_a_cnt += 1

                if is_a_cnt == 2:
                    is_a_found = True

                if not found_subj or is_a_found:
                    event_tokens = self.make_event_tokens_list(parent_node, event_tokens)
                event_tokens = list(set(event_tokens))
                event_tokens = sorted(event_tokens, key=lambda x: x[1])
                event = " ".join(
                    [
                        self.event_matcher.morph_parse(tok)
                        for tok, tok_ord in event_tokens
                        if tok.lower() not in self.stopwords
                        and tok.lower() not in date_str_tokens
                        and not tok.lower().startswith("год")
                    ]
                )
                candidate_events, candidate_dates, scores = self.event_matcher.search(event)

        return candidate_events, candidate_dates, scores

    def check_date(self, tree, date):
        date_info, date_str = date
        match = True
        if date_info:
            candidate_events, candidate_dates, scores = self.link_event(tree, date)
            if candidate_dates and scores[0] > self.tfidf_dot_product_thres:
                num_mismatches = 0
                check_range = min(len(candidate_dates), 5)
                for i in range(check_range):
                    current_match = True
                    for candidate_date in candidate_dates[i]:
                        if "date" in date_info.keys():
                            if "date" in candidate_date.keys():
                                for date_elem in date_info["date"].keys():
                                    if candidate_date["date"].get(date_elem, 0) != date_info["date"][date_elem]:
                                        current_match = False
                                        break
                                if not current_match:
                                    num_mismatches += 1
                                    break
                            elif "date_interval" in candidate_date.keys():
                                for date_elem in date_info["date"].keys():
                                    if (
                                        candidate_date["date_interval"]["start"].get(date_elem, 0)
                                        != date_info["date"][date_elem]
                                    ):
                                        current_match = False
                                        break
                                if not current_match:
                                    num_mismatches += 1
                                    break
                            else:
                                num_mismatches += 1
                                break

                        if "date_interval" in date_info.keys():
                            if "date_interval" in candidate_date.keys():
                                for date_elem in date_info["date_interval"]["start"].keys():
                                    if (
                                        candidate_date["date_interval"]["start"].get(date_elem, 0)
                                        != date_info["date_interval"]["start"][date_elem]
                                    ):
                                        current_match = False
                                        break
                                for date_elem in date_info["date_interval"]["end"].keys():
                                    if (
                                        candidate_date["date_interval"]["end"].get(date_elem, 0)
                                        != date_info["date_interval"]["end"][date_elem]
                                    ):
                                        current_match = False
                                        break
                                if not current_match:
                                    num_mismatches += 1
                                    break
                            else:
                                num_mismatches += 1
                                break
                if num_mismatches >= check_range:
                    match = False
        return match

    def make_event_tokens_list(self, node, event_tokens_list):
        event_tokens_list.append((node.form, node.ord))
        for elem in node.children:
            if elem.deprel not in ["acl:relcl", "acl", "conj"] or (
                elem.deprel == "acl" and elem.form.startswith("получивш")
            ):
                event_tokens_list = self.make_event_tokens_list(elem, event_tokens_list)

        return event_tokens_list

    def find_date(self, sentence):
        month_dict = {
            "янв": 1,
            "фев": 2,
            "мар": 3,
            "апр": 4,
            "май": 5,
            "мая": 5,
            "мае": 5,
            "июн": 6,
            "июл": 7,
            "авг": 8,
            "сен": 9,
            "окт": 10,
            "ноя": 11,
            "дек": 12,
        }
        date_templates_regexp = [
            "с ([\d]{1,2}) (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря) "
            + "([\d]{3,4}) года по ([\d]{1,2}) (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|"
            + "октября|ноября|декабря) ([\d]{3,4})",
            "с ([\d]{1,2}) (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря) "
            + "([\d]{3,4}) по ([\d]{1,2}) (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|"
            + "октября|ноября|декабря) ([\d]{3,4})",
            "с (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря) "
            + "([\d]{3,4}) года по (январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|"
            + "ноябрь|декабрь) ([\d]{3,4})",
            "с (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря) ([\d]{3,4}) "
            + "по (январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь) ([\d]{3,4})",
            "([\d]{1,2}) (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря) "
            + "([\d]{3,4})",
            "в (январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре) ([\d]{3,4})",
            "(январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь) ([\d]{3,4})",
            "([\d]{3,4})-е годы",
            "([\d]{3,4})-x годах",
            "([\d]{3,4})-([\d]{3,4})",
            "([\d]{3,4}) - ([\d]{3,4})",
            "с ([\d]{3,4}) по ([\d]{3,4})",
            "([\d]{3,4}) г\.",
            "([\d]{3,4}) г ",
            "([\d]{3,4})",
        ]
        date_templates = [
            "c {} {} {} года по {} {} {}",
            "c {} {} {} по {} {} {}",
            "c {} {} года по {} {}",
            "c {} {} по {} {}",
            "{} {} {}",
            "в {} {}",
            "{} {}",
            "{}-е годы",
            "{}-x годах",
            "{}-{}",
            "{} - {}",
            "с {} по {}",
            "{} г.",
            "{} г",
            "{}",
        ]
        dates = []
        date_info = {}
        date_str = ""

        for n, date_template in enumerate(date_templates_regexp):
            fnd = re.findall(date_template, sentence)
            if fnd:
                # c 11 января 1900 года по 12 февраля 1901      # c 11 января 1900 по 12 февраля 1901
                if n == 0 or n == 1:
                    for elem in fnd:
                        date_info = {}
                        date_info["date_interval"] = {
                            "start": {"year": elem[2], "month": month_dict[elem[1][:3]], "day": elem[0]},
                            "end": {"year": elem[5], "month": month_dict[elem[4][:3]], "day": elem[3]},
                        }
                        date_str = date_templates[n].format(*elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # c января 1900 года по февраля 1901        # c января 1900 по февраля 1901
                if n == 2 or n == 3:
                    for elem in fnd:
                        date_info = {}
                        date_info["date_interval"] = {
                            "start": {"year": elem[1], "month": month_dict[elem[0][:3]]},
                            "end": {"year": elem[3], "month": month_dict[elem[2][:3]]},
                        }
                        date_str = date_templates[n].format(*elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # 11 января 1900
                if n == 4:
                    for elem in fnd:
                        date_info = {}
                        date_info["date"] = {"year": elem[2], "month": month_dict[elem[1][:3]], "day": elem[0]}
                        date_str = date_templates[n].format(*elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # в январе 1900     # январь 1900
                if n == 5 or n == 6:
                    for elem in fnd:
                        date_info = {}
                        date_info["date"] = {"year": elem[1], "month": month_dict[elem[0][:3]]}
                        date_str = date_templates[n].format(*elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # 1900-е годы       # 1990-x годах
                if n == 7 or n == 8:
                    for elem in fnd:
                        date_info = {}
                        date_info["decade"] = {"value": elem}
                        date_str = date_templates[n].format(elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # 1900-1901     # 1900 - 1901       # с 1900 по 1901
                if n == 9 or n == 10 or n == 11:
                    for elem in fnd:
                        date_info = {}
                        date_info["date_interval"] = {"start": {"year": elem[0]}, "end": {"year": elem[1]}}
                        date_str = date_templates[n].format(*elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break
                # 1900 г.   # 1900 г    # 1900
                if n == 12 or n == 13 or n == 14:
                    for elem in fnd:
                        date_info = {}
                        date_info["date"] = {"year": elem}
                        date_str = date_templates[n].format(elem)
                        sentence = sentence.replace(date_str, "")
                        dates.append((date_info, date_str))
                    break

        return dates
