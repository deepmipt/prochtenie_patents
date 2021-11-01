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

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 3000))
DEEPPAVLOV_CONFIG = os.getenv("DEEPPAVLOV_CONFIG")
STORE_DATA_ENABLE = bool(os.getenv("STORE_DATA_ENABLE", False))
INPUT_DATA_FILE = "server_input_data.jsonl"


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")

parser = build_model(DEEPPAVLOV_CONFIG)


def handler(annotations):
    """
    Calculates the annotation of the format
    [[labels_11, labels_12, ...], ...]
    for sentence data in format
    [[sent_11, sent_12, ...], ...]

    Each labels_ij is a list of NER labels for a given tokenized sentence.
    """
    sents = annotations["annotations"]["basic_reader"]["clear_essay_sentences"]
    parsed_tokens = [[None] * len(elem) for elem in sents]
    parsed_labels = [[None] * len(elem) for elem in sents]
    data_to_parse = list(elem["words"] for elem in itertools.chain(*sents))
    logger.info(f"sentences: {str(data_to_parse)}")
    tokens, labels = parser.batched_call(data_to_parse)
    index = 0
    for i, paragraph in enumerate(sents):
        for j, _ in enumerate(paragraph):
            parsed_tokens[i][j] = tokens[index]
            parsed_labels[i][j] = labels[index]
            index += 1
    return {"ner_labels": parsed_labels, "ner_tokens": parsed_tokens}


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
