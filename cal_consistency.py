from scipy.stats import spearmanr, rankdata, kendalltau
import statistics
import numpy as np
import os

from tools import tools_json_load, tools_log_on_rank, tools_json_dump, tools_get_time
from config import ConsistencyMethodEnum, CalConsistencyConfig, parse_args


if __name__ == '__main__':
    args: CalConsistencyConfig = parse_args(CalConsistencyConfig, pass_in=[])

    # load data
    file1 = tools_json_load(args.file1)
    file2 = tools_json_load(args.file2)

    # overlap data
    keys = set(file1.keys()) & set(file2.keys())

    # calculate consistency
    consistency_rates = []
    for key in keys:
        score1 = file1[key]['reward']
        score2 = file2[key]['reward']

        if len(score1) != len(score2) or \
            any(i == None for i in score1) or any(i == None for i in score2):
            tools_log_on_rank(f"key={key} has different length of scores or None scores, skipped", level='warning')
            continue
        
        rank1 = rankdata(score1, method='ordinal')
        rank2 = rankdata(score2, method='ordinal')

        # cal consistency rate
        if args.method == ConsistencyMethodEnum.kendall:
            consistency, _ = kendalltau(rank1, rank2)
            # scale from [-1, 1] to [0, 1]
            consistency = (consistency + 1) / 2

        elif args.method == ConsistencyMethodEnum.spearman:
            consistency, _ = spearmanr(rank1, rank2)
            consistency = (consistency + 1) / 2

        elif args.method == ConsistencyMethodEnum.toporder:
            if rank1[0] == rank2[0] and rank1[-1] == rank2[-1]:
                consistency = 1.0
            else:
                consistency = 0.0

        else:
            raise ValueError(f"method={args.method} not supported")

        consistency_rates.append(consistency)

    # statistics
    tools_log_on_rank(args)

    mean = statistics.mean(consistency_rates)
    min = min(consistency_rates)
    max = max(consistency_rates)
    std = statistics.stdev(consistency_rates)

    if args.method == ConsistencyMethodEnum.kendall:
        tools_log_on_rank("Note that the kendall tau (consistency rate) has been scaled from [-1, 1] to [0, 1]", level='warning')

    tools_log_on_rank(f"Number of file1={len(file1)}, file2={len(file2)}, total valid overlap={len(consistency_rates)}\n{args.method.name} consistency rate\nmean={mean}, min={min}, max={max}, std={std}")

    # generate dpo training data
    tools_log_on_rank(f"file1={args.file1} will be used to generate dpo training data")

    key2consitency = {key: consistency for key, consistency in zip(keys, consistency_rates)}
    results = {}
    for k, item in file1.items():
        selected_idx = np.argmax(item['reward'])
        rejected_idx = np.argmin(item['reward'])

        if selected_idx == rejected_idx:
            tools_log_on_rank(f"key={k} has the same selected and rejected index, skipped", level='warning')
            continue

        results[k] = {
            'prompt': item['prompt'],
            'selected': item['response'][selected_idx],
            'rejected': item['response'][rejected_idx],
            'consistency': key2consitency.get(k)
        }
    
    if args.output is None:
        args.output = f"{args.file1.removesuffix('.json')}.{args.method.name}.dpo.json"
    
    tools_json_dump(results, args.output)

    tools_log_on_rank(f"file1 dpo training data is saved to={args.output}")
    
