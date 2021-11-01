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
from itertools import chain

from cp_data_readers import prochtenie_reader
from cp_data_store import store
from flask import Flask, jsonify, request
from healthcheck import HealthCheck
from preprocess import find_dupl_inds, morph_parse
from sacremoses import MosesDetokenizer

md = MosesDetokenizer(lang="ru")

SERVICE_NAME = os.getenv("SERVICE_NAME", "dupl_error_solver")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 2090))
STORE_DATA_ENABLE = bool(os.getenv("STORE_DATA_ENABLE", False))
INPUT_DATA_FILE = "server_input_data.jsonl"


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")

# clear old file
# store.rm_file(INPUT_DATA_FILE)


def handler(instance):
    subject = instance["instance_info"]["subject"]
    if subject not in ["русский язык", "литература", "история", "обществознание"]:
        return {"mistakes": []}
    parsed = list(chain.from_iterable(instance["annotations"]["morphosyntactic_parser"]["annotated_sentences"]))
    parsed.append([["END", "END", "END", "END", "END", "END", "END", "END", "END", "END"]])
    words, lemmas, pos = morph_parse(parsed)
    dupl_inds = find_dupl_inds(words, lemmas, pos)
    tags = {}
    tag_counter = 1
    if any(chain.from_iterable(dupl_inds)):
        res_sents = []
        out_dupls_inds = set()
        for i, (inner, outer) in enumerate(dupl_inds):
            out_first = set(chain.from_iterable([x for x, _ in outer]))
            dupls_inds = set(inner) | out_first | out_dupls_inds
            sent = []
            for j, x in enumerate(words[i]):
                if j in dupls_inds:
                    lemma = lemmas[i][j]
                    if lemma not in tags:
                        tags[lemma] = tag_counter
                        tag_counter += 1
                    sent.append(f"(\\Р.повтор\\ {x} #{tags[lemma]} \)")
                else:
                    sent.append(x)
            res_sents.append(sent)
            out_dupls_inds = set(chain.from_iterable([x for _, x in outer]))
        detokenized_sents = [md.tokenize(x) for x in res_sents]
        res = " ".join([x for x in detokenized_sents])
        _, mistakes = prochtenie_reader.parse_essay(res, subject)
        return mistakes
    else:
        res = " ".join([md.detokenize(" ".join([x for x in w])) for w in words])
        _, mistakes = prochtenie_reader.parse_essay(res, subject)
        return mistakes


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
