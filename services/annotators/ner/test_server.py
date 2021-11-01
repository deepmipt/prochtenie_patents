# flake8: noqa
#
##########################################################################
# Attention, this file cannot be changed, if you change it I will find you#
##########################################################################
#
import argparse
import json
import os
import time

import requests
from cp_tests import utils

SEED = 31415
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 3000))
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknow_skill")
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "test_data")


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rewrite_ground_truth", action="store_true", default=False)
args = parser.parse_args()


def test_skill(rewrite_ground_truth):
    url = f"http://0.0.0.0:{SERVICE_PORT}/model"
    warnings = 0

    for request_file, response_file in utils.get_data(TEST_DATA_DIR):
        request = json.load(request_file.open())

        create_response_file_flag = not response_file.exists()
        st_time = time.time()
        response = requests.post(url, json=request, timeout=180).json()[0]
        total_time = time.time() - st_time
        print(f"exec time: {total_time:.3f}s")
        if create_response_file_flag or rewrite_ground_truth:
            json.dump(response, response_file.open("wt"), ensure_ascii=False, indent=4)
            uid, gid = request_file.stat().st_uid, request_file.stat().st_gid
            os.chown(str(response_file), uid, gid)
            is_equal_flag, msg = False, "New output file is created"
        else:
            is_equal_flag, msg = utils.compare_structs(json.load(response_file.open()), response)
        if not is_equal_flag:
            print("----------------------------------------")
            print(f"cand = {response}")
            print(msg)
            print(f"request_file = {request_file}")
            print(f"response_file = {response_file}")
            warnings += 1
    assert warnings == 0
    print("SUCCESS!")


if __name__ == "__main__":
    test_skill(args.rewrite_ground_truth)
