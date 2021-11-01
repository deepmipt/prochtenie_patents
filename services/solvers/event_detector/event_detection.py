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

import re
from string import punctuation

import nltk
from deeppavlov.core.common.registry import register


@register("event_detector")
class EventDetector:
    def __init__(self, ner, ner_parser, **kwargs):
        self.ner = ner
        self.ner_parser = ner_parser

    def __call__(self, sentences, offsets):
        events = []
        chunks = []
        chunk = ""
        cur_len = 0
        for sentence in sentences:
            sentence_tokens = nltk.word_tokenize(sentence)
            if cur_len + len(sentence_tokens) < 300:
                chunk += f"{sentence} "
                cur_len += len(sentence_tokens)
            else:
                chunks.append(chunk.strip())
                chunk = f"{sentence} "
        if chunk:
            chunks.append(chunk)

        tokens, probas = self.ner(chunks)
        events_batch, _, _ = self.ner_parser(tokens, probas)
        events_list = [
            self.delete_punctuation(event_substr) for events_list in events_batch for event_substr in events_list
        ]

        for sentence, offset in zip(sentences, offsets):
            sanitized_sentence = self.delete_punctuation(sentence)
            for event_substr in events_list:
                if event_substr in sanitized_sentence:
                    start_pos = offset[0]
                    end_pos = start_pos + len(sentence)
                    events.append({"span_start": start_pos, "span_end": end_pos, "type": "событие", "text": sentence})
                    break

        return events

    def delete_punctuation(self, text):
        sanitized_text = "".join([ch for ch in text if ch not in punctuation])
        sanitized_text = re.sub(" +", " ", sanitized_text)
        return sanitized_text

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
