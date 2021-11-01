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

from cp_data_readers import json_prochtenie_reader, neznaika_reader
from cp_data_store import store
from flask import Flask, jsonify, request
from healthcheck import HealthCheck

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 3000))
STORE_DATA_ENABLE = bool(os.getenv("STORE_DATA_ENABLE", False))
INPUT_DATA_FILE = "server_input_data.jsonl"


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")

# clear old file
# store.rm_file(INPUT_DATA_FILE)


def handler(instance):
    raw_input = instance["raw_input"]
    if "neznaika" == instance["instance_info"]["dataset_name"]:
        subject = instance["instance_info"]["subject"]
        file_name = instance["instance_info"]["file_name"]
        parsed_text = neznaika_reader.parse_text(raw_input, subject=subject, file_name=file_name)
    elif "prochtenie" == instance["instance_info"]["dataset_name"]:
        parsed_text = json_prochtenie_reader.parse_text(raw_input)
    else:
        raise "not implemented"
    return parsed_text


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

    responses = [handler(instance) for instance in request.json["input_data"]]

    total_time = time.time() - st_time
    logger.info(f"{SERVICE_NAME} exec time: {total_time:.3f}s")
    return jsonify(responses)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)
