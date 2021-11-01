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

import itertools
import logging
import os
import time

from cp_data_store import store
from deeppavlov import build_model
from flask import Flask, jsonify, request
from healthcheck import HealthCheck
from model import detect_roles

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 2094))
DEEPPAVLOV_CONFIG = os.getenv("DEEPPAVLOV_CONFIG")
STORE_DATA_ENABLE = bool(os.getenv("STORE_DATA_ENABLE", False))
INPUT_DATA_FILE = "server_input_data.jsonl"

MIN_PERSON_COUNT = int(os.getenv("min_person_count", 1))


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")

role_classifier = build_model("hist_roles_classification.json", download=True)


def handler(annotations):
    """
    
    """
    if annotations["annotations"]["basic_reader"]["subject"].lower() != "история":
        return {}
    # preparing the input
    sents = list(
        elem["text"]
        for elem in itertools.chain.from_iterable(annotations["annotations"]["basic_reader"]["clear_essay_sentences"])
    )
    ner_labels = list(itertools.chain.from_iterable(annotations["annotations"]["ner"]["ner_labels"]))
    parses = list(
        itertools.chain.from_iterable(annotations["annotations"]["morphosyntactic_parser"]["annotated_sentences"])
    )
    offsets = list(
        itertools.chain.from_iterable(annotations["annotations"]["basic_reader"]["clear_essay_word_offsets"])
    )
    # obtaining the roles
    roles_data = detect_roles(
        parses, ner_labels, sents=sents, min_person_count=MIN_PERSON_COUNT, return_sentence_indexes=True
    )
    # postprocessing the data

    sentences_with_roles = []

    for person, sentence_indexes in roles_data.items():
        for sent_index, _ in sentence_indexes:
            sentences_with_roles.append(sents[sent_index])

    k2_points = 0

    for paragraph in annotations["annotations"]["basic_reader"]["clear_essay_sentences"]:
        paragraph_text = ""
        for sentence in paragraph:
            paragraph_text += f"{sentence} "
        for sentence in sentences_with_roles:
            if sentence in paragraph_text:
                pass
        res = role_classifier([paragraph_text])
        if res[0][1] > res[0][0]:
            k2_points += 1

    if k2_points > 2:
        k2_points = 2

    if len(roles_data) == 1 and k2_points == 2:
        k2_points = 1
    if len(roles_data) >= 3:
        k2_points = 2

    at_least_one_full_role = False
    for role in roles_data:
        if len(roles_data[role]) > 1:
            at_least_one_full_role = True
            break

    if at_least_one_full_role and k2_points == 0:
        k2_points = 1

    return {"criteria": {"к2": k2_points}}


@app.route("/model", methods=["POST"])
def respond():
    """A handler of requests.
    To use:
    curl -X POST "http://localhost:${PORT}/model" \
    -H "accept: application/json"  -H "Content-Type: application/json" \
    -d "{ \"args\": [ \"data\" ]}"
    """
    st_time = time.time()

    if STORE_DATA_ENABLE:
        store.save2json_line(request.json, INPUT_DATA_FILE)
    # `instance` is a single document
    responses = [None] * len(request.json["input_data"])
    logger.info(f"{SERVICE_NAME}: parsing {len(responses)} sentences.")
    for i, elem in enumerate(request.json["input_data"]):
        responses[i] = handler(elem)
    total_time = time.time() - st_time
    logger.info(f"{SERVICE_NAME} exec time: {total_time:.3f}s")
    return jsonify(responses)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)
