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

import logging
import os
import time

from cp_data_store import store
from flask import Flask, jsonify, request
from healthcheck import HealthCheck
from model import check_cause_effect

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 2102))
DEEPPAVLOV_CONFIG = os.getenv("DEEPPAVLOV_CONFIG")
STORE_DATA_ENABLE = bool(os.getenv("STORE_DATA_ENABLE", False))
INPUT_DATA_FILE = "server_input_data.jsonl"

MIN_PERSON_COUNT = int(os.getenv("min_person_count", 1))


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")


def handler(annotations):
    """
    
    """
    if annotations["annotations"]["basic_reader"]["subject"].lower() != "история":
        return {}

    if annotations["instance_info"]["dataset_name"] == "neznaika":
        topic = annotations["annotations"]["basic_reader"]["header"]
    else:
        topic = annotations["annotations"]["basic_reader"]["meta"]["тема"]

    paragraphs = annotations["annotations"]["basic_reader"]["clear_essay_sentences"]
    k3_points = check_cause_effect(paragraphs, topic)

    return {"criteria": {"к3": k3_points}}


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
