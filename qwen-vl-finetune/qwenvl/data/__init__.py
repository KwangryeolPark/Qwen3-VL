import os
import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

# GroundCUA GUI-grounding SFT data prepared by tools/prepare_groundcua_sft.py.
# Absolute image paths are stored in the JSONL itself, so data_path can remain empty.
# The environment variable makes the fork portable across machines while preserving a
# sensible default for this project's layout.
GROUND_CUA_SFT = {
    "annotation_path": os.environ.get(
        "GROUND_CUA_SFT_JSONL",
        "/home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/sft/qwen3vl_train.jsonl",
    ),
    "data_path": "",
}

GROUND_CUA_SFT_SMOKE = {
    "annotation_path": os.environ.get(
        "GROUND_CUA_SFT_SMOKE_JSONL",
        "/home/kwangryeol/workspace/Qwen3-8B-Instruct/datasets/groundcua/sft/qwen3vl_train_smoke.jsonl",
    ),
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "groundcua_sft": GROUND_CUA_SFT,
    "groundcua_sft_smoke": GROUND_CUA_SFT_SMOKE,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
