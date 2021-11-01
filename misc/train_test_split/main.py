# %%

import json
import pathlib
import collections

import sklearn

from cp_data_readers.map_update import SUBJECT_LIST
from cp_data_readers import neznaika_reader


# %%

reduction2subject = {
    "rus": "русский язык",
    "lit": "литература",
    "his": "история",
    "hist": "история",
    "soc": "обществознание",
    "obch": "обществознание",
    "eng": "английский",
}

subject2reduction = {
    "русский язык": "rus",
    "литература": "lit",
    "история": "his",
    "обществознание": "soc",
    "английский": "eng",
}

for sub in reduction2subject.values():
    assert sub in SUBJECT_LIST


def create_instance_with_metadata(file):
    try:
        raw_input = file.open().read()
    except Exception:
        raw_input = file.open(encoding="windows-1251").read()
    if not raw_input:
        return

    try:
        if "neznaika" == file.parent.parent.name:
            instance = {
                "raw_input": json.loads(raw_input, encoding="utf8"),
                "instance_info": {},
            }
            instance["instance_info"]["dataset_split"] = None
            instance["instance_info"]["dataset_name"] = "neznaika"
            instance["instance_info"]["dataset_version"] = "v2"
            instance["instance_info"]["subject"] = reduction2subject[file.parent.name]

        elif "prochtenie" == file.parent.parent.name:
            raise "not implemented"
        else:
            raise
    except Exception:
        print(file)
        return {}
    return {
        "file": file,
        "instance": instance,
    }


def get_criteria(instance_with_metadata):
    instance = instance_with_metadata["instance"]
    raw_input = instance["raw_input"]
    if instance["instance_info"]["dataset_name"] == "neznaika":
        subject = instance["instance_info"]["subject"]
        parsed_text = neznaika_reader.parse_text(raw_input, subject=subject, file_name="")

    elif "prochtenie" == instance["instance_info"]["dataset_name"]:
        raise "not implemented"
    else:
        raise
    instance_with_metadata["criteria"] = parsed_text["criteria"]
    return instance_with_metadata


def split_instances(instances):
    subject_instances = collections.defaultdict(list)
    for instance in instances:
        subject_instances[instance["instance"]["instance_info"]["subject"]] += [instance]
    criteria_keys = [f"k{i}" for i in range(1, 13)]
    splitted_subject_instances = {}
    for subject in subject_instances.keys():

        instances, labels = zip(
            *[
                (instance, [instance["criteria"].get(key, 0) for key in criteria_keys])
                for instance in subject_instances[subject]
            ]
        )
        instances_train, instances_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
            instances, labels, test_size=0.25, stratify=labels, random_state=315
        )
        print(f"========{subject}========")
        print(f"instances_train size = {len(instances_train)}")
        print(f"instances_test size = {len(instances_test)}")
        print("")
        for instance in instances_train:
            instance["instance"]["instance_info"]["dataset_split"] = "train"
        for instance in instances_test:
            instance["instance"]["instance_info"]["dataset_split"] = "test"
        splitted_subject_instances[subject] = {
            "train": instances_train,
            "test": instances_test,
        }
    return splitted_subject_instances


output_dataset_dir = pathlib.Path("../../venv/datasets/neznaika.v2/")


def save_dataset(splitted_subject_instances):
    for subject in splitted_subject_instances.keys():
        subject_reduction = subject2reduction[subject]

        for dataset_split in splitted_subject_instances[subject].keys():
            dataset_split_dir = output_dataset_dir / subject_reduction / dataset_split
            dataset_split_dir.mkdir(parents=True, exist_ok=True)
            for instance in splitted_subject_instances[subject][dataset_split]:
                output_file = dataset_split_dir / instance["file"].name
                json.dump(instance["instance"], output_file.open("wt"), ensure_ascii=False, indent=4)


# %%
curr_data_path = pathlib.Path("../../data/datasets/neznaika")
files = list(curr_data_path.glob("./**/*.json"))
len(files)
# %%

instances = [create_instance_with_metadata(file) for file in files]
# %%

instances = [instance for instance in instances if instance]
instances = [get_criteria(instance) for instance in instances]

# %%
splitted_subject_instances = split_instances(instances)
# %%
save_dataset(splitted_subject_instances)
