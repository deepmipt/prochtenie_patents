# %%

import json
import pathlib
import copy
import collections
import argparse


def is_merged_selections(selection1, selection2):
    left_edge = min(selection1["startSelection"], selection2["startSelection"])
    right_edge = max(selection1["endSelection"], selection2["endSelection"])
    gap_length = right_edge - left_edge
    leght1 = selection1["endSelection"] - selection1["startSelection"]
    leght2 = selection2["endSelection"] - selection2["startSelection"]
    return gap_length <= leght1 + leght2


def get_merged_indexes(selections):
    for index1, sel1 in enumerate(selections):
        for index2, sel2 in enumerate(selections):
            if index1 != index2 and is_merged_selections(sel1, sel2):
                return [index1, index2]


def merge_selections(selections, add_cross):
    start_selection = min([sel["startSelection"] for sel in selections])
    end_selection = max([sel["endSelection"] for sel in selections])
    merged_selection = selections[-1]
    merged_selection["startSelection"] = start_selection
    merged_selection["endSelection"] = end_selection
    if add_cross:
        merged_selection["crossed"] = True
    return merged_selection


def make_union(*input_data, add_cross=False):
    selections = sum([data.get("selections", []) for data in input_data], [])
    error_types2selections = collections.defaultdict(list)
    for sel in selections:
        error_types2selections[sel["type"]] += [sel]
    for error_type in error_types2selections.keys():
        typed_selections = error_types2selections[error_type]
        prev_len = 0
        while prev_len != len(typed_selections):
            prev_len = len(typed_selections)
            merged_indexes = get_merged_indexes(typed_selections)
            if merged_indexes is not None:
                merged_selections = [typed_selections.pop(index) for index in sorted(merged_indexes, reverse=True)]
                merged_selection = merge_selections(merged_selections, add_cross)
                typed_selections.append(merged_selection)

        error_types2selections[error_type] = typed_selections
    data = copy.deepcopy(input_data[-1])
    data["selections"] = sum(error_types2selections.values(), [])
    return data


def make_intersection(*input_data):
    data_union = make_union(*input_data, add_cross=True)
    selections = [sel for sel in data_union["selections"] if sel.get("crossed")]
    for sel in selections:
        del sel["crossed"]
    data_union["selections"] = selections
    return data_union


def run_cmd(args):
    union_dir = args.out_datasetes_dir / "union_prochtenie"
    intersection_dir = args.out_datasetes_dir / "intersection_prochtenie"
    union_dir.mkdir(parents=True, exist_ok=True)
    intersection_dir.mkdir(parents=True, exist_ok=True)
    tasks2files = collections.defaultdict(list)
    for file in args.in_dataset_dir.glob("./*.json"):
        name = file.stem
        if "_" in name:
            tasks2files[name.split("_")[0]] += [file]
    for task_id, files in tasks2files.items():
        input_data = [json.load(file.open()) for file in files]
        union_data = make_union(*input_data)
        json.dump(union_data, (union_dir / f"{task_id}.json").open("wt"), indent=4, ensure_ascii=False)
        intersection_data = make_intersection(*input_data)
        json.dump(intersection_data, (intersection_dir / f"{task_id}.json").open("wt"), indent=4, ensure_ascii=False)


#  misc/merge_prochtenie/main.py --in_dataset_dir data/datasets/prochtenie_eng_train --out_datasetes_dir data/datasets
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dataset_dir",
        help="A dataset dir",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--out_datasetes_dir",
        help="A dataset dir",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    run_cmd(args)
