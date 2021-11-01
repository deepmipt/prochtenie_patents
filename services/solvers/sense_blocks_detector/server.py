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
from deeppavlov import build_model
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

model = build_model("custom_config.json")


def create_section(meaning, block_id, start_offset, end_offset):
    assert end_offset is not None
    meaning = "АРГУМЕНТ" if meaning == "ДОВОД" else meaning
    section = {
        "comment": "",
        "correction": "",
        "endSelection": end_offset,
        "explanation": "",
        "group": "meaning",
        "id": block_id,
        "startSelection": start_offset,
        "subtype": "",
        "tag": "",
        "type": meaning,
    }
    return section


def handler(instance):
    if instance["instance_info"]["subject"] not in ["eng"]:
        return {"selections": []}

    sentences = instance["annotations"]["basic_reader"]["extended_markup"]["clear_essay_sentences"]
    sentence_offsets = instance["annotations"]["basic_reader"]["extended_markup"]["clear_essay_word_offsets"]

    block_id = 0  # ids of blocks
    sections = []
    label = ""  # current label
    flushed = True  # for adding last section
    start_offset = 0  # starting offset of meaning block
    end_offset = None

    for block, block_offset in zip(sentences, sentence_offsets):
        sentences = [sent["text"] for sent in block]
        if not sentences:
            continue
        offsets = [(sent[0], sent[-1]) for sent in block_offset]
        predictions = model(sentences)

        if not flushed:  # end at each block - flush (end) current meaning block
            if label != "":
                sections.append(create_section(label, block_id, start_offset, end_offset))
            block_id += 1
            label = ""
            flushed = True
            start_offset = offsets[0][0]

        for pred, offset in zip(predictions, offsets):
            if pred != "random":
                is_arg = label == "ДОВОД" or label == "АРГУМЕНТ"
                is_arg = is_arg and (pred == "ДОВОД" or pred == "АРГУМЕНТ")
                if is_arg:  # merge consecutive arguments
                    continue
                flushed = False
                if label != "":
                    sections.append(create_section(label, block_id, start_offset, end_offset))
                    block_id += 1
                label = pred
                start_offset = offset[0]
            end_offset = offset[1]
    if not flushed:
        sections.append(create_section(label, block_id, start_offset, end_offset))
    return {"selections": sections}


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
    logger.info(responses)
    logger.info(f"{SERVICE_NAME} exec time: {total_time:.3f}s")
    return jsonify(responses)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)
