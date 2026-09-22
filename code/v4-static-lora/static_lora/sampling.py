"""Frozen training exposure queues used to produce the existing LoRA checkpoints."""
from collections import defaultdict, deque
import random


def stratified_order(records, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record["category"]].append(i)
    for group in groups.values():
        rng.shuffle(group)
    names = sorted(groups)
    rng.shuffle(names)
    queues = {name: deque(groups[name]) for name in names}
    result = []
    while any(queues.values()):
        for name in names:
            if queues[name]:
                result.append(queues[name].popleft())
    return result


class Queue:
    def __init__(self, records, seed):
        self.records, self.seed, self.cursor = records, seed, 0
        self._epoch, self._order = -1, None

    def next(self):
        epoch, index = divmod(self.cursor, len(self.records))
        if self._epoch != epoch:
            self._epoch = epoch
            self._order = stratified_order(self.records, self.seed + epoch)
        self.cursor += 1
        return self._order[index]


class ExposurePlan:
    """Persistent ordinary/temporal queues and mode coverage; exactly 6:2 each update."""
    def __init__(self, records, seed):
        self.records, self.seed = records, seed
        self.ordinary, self.temporal = Queue(records, seed), Queue(records, seed + 7919)
        self.counts = [dict(i2v=0, t2v=0, temp=0) for _ in records]

    def batch(self, step):
        rng = random.Random(self.seed + step * 31)
        samples = [(self.ordinary.next(), False) for _ in range(7)]
        samples.append((self.temporal.next(), True))
        # Fix the temporal 3:1 ratio in every four consecutive JOINT batches.
        offset = (step - 1) % 4
        cycle = (step - 1) // 4
        temporal_t2v = random.Random(self.seed + 100003 + cycle).randrange(4) == offset
        modes = {7: "t2v" if temporal_t2v else "i2v"}
        candidates = list(range(7))
        rng.shuffle(candidates)
        # Higher deficit means a video has received too few T2V exposures; rotate eligible videos.
        candidates.sort(key=lambda i: (self.counts[samples[i][0]]["i2v"] -
                                       3 * self.counts[samples[i][0]]["t2v"]), reverse=True)
        n_t2v = 2 - sum(mode == "t2v" for mode in modes.values())
        for rank, i in enumerate(candidates):
            modes[i] = "t2v" if rank < n_t2v else "i2v"
        results = []
        for i, (index, temporal) in enumerate(samples):
            counts = self.counts[index]
            exposure = counts["i2v"] + counts["t2v"]
            results.append(dict(index=index, mode=modes[i], temporal=temporal,
                                repair=temporal and step >= 801, exposure=exposure))
            counts[modes[i]] += 1
            counts["temp"] += int(temporal)
        rng.shuffle(results)
        return results

    def state_dict(self):
        return dict(ordinary=self.ordinary.cursor, temporal=self.temporal.cursor, counts=self.counts)

    def load_state_dict(self, state):
        self.ordinary.cursor, self.temporal.cursor = state["ordinary"], state["temporal"]
        self.counts = state["counts"]
