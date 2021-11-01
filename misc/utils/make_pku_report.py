# %%

import json
import pathlib
import collections
import argparse

# %%
def run_cmd(args):
    markuped_tasks = [json.loads(line) for line in args.jsonl_file.open().readlines()]
    task2markups = {
        data["essayId"]: [
            {
                "STER_AVG": data["STER_AVG"],
                **markup,
            }
            for markup in data["markups"]
            if not markup["isExp"]
        ][-1]
        for data in markuped_tasks
    }
    matches = sum([markup["matching"] for markup in task2markups.values()], [])
    metrics = collections.defaultdict(list)
    for match in matches:
        for metric_label, metric_value in match["metrics"].items():
            metrics[metric_label] += [metric_value]

    avr_metrics = {
        metric_label: round(sum(metric_values, 0) / len(metric_values), 2)
        for metric_label, metric_values in metrics.items()
    }
    avr_otar = [markup["OTAR"] for markup in task2markups.values()]
    avr_otar = sum(avr_otar, 0) / len(avr_otar)
    avr_star = [markup["STAR"] for markup in task2markups.values()]
    avr_star = sum(avr_star, 0) / len(avr_star)
    avr_ster = [markup["STER_AVG"] for markup in task2markups.values()]
    avr_ster = sum(avr_ster, 0) / len(avr_ster)
    report = {
        "averages": {
            "metrics": avr_metrics,
            "OTAR": avr_otar,
            "STAR": avr_star,
            "STER": avr_ster,
        },
        "tasks": task2markups,
    }
    json.dump(report, args.report_file.open("wt"), ensure_ascii=False, indent=4)
    print(f"averages = {report['averages']}")
    print(f"report_dir = {args.report_file.parent.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--jsonl_file", help="jsonl file", type=pathlib.Path)
    parser.add_argument("-o", "--report_file", help="report file", type=pathlib.Path)
    args = parser.parse_args()
    run_cmd(args)
