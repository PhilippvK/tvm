def estimate_size(history):
    history = list(map(int, history))
    assert len(history) > 0
    task_trials = int(sum(history))
    last_num = history[-1]
    # print("last_num", last_num)
    if last_num == 0:
        return task_trials, False
    else:
        num_batches = len(history)
        batch_idxs = list(range(num_batches))
        if num_batches < 3:
            return int(task_trials * 1.1), True
        else:
            ratios = []
            for prev, curr in zip(history[:-1], history[1:]):
                if prev > 0:
                    ratios.append(curr / prev)
            recent_ratios = ratios[-3:]
            ratio = sum(recent_ratios) / len(recent_ratios)
            # print("ratios", ratios)
            # print("recent_ratios", recent_ratios)
            # print("avg_ratio", ratio)
            ratio = max(0.0, min(ratio, 0.999))
            if ratio >= 0.95:
                # Conservative fallback
                estimated_remaining = last_num * 10.0
            else:
                # Infinite geometric tail:
                #
                # last*r + last*r^2 + ...
                #
                estimated_remaining = last_num * ratio / (1.0 - ratio)
            # print("estimated_remaining", estimated_remaining)
            estimated_total = task_trials + estimated_remaining

            return int(estimated_total), True
