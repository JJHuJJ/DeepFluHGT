# METRIC-DIR-001 unit test: window-level direction_accuracy fix (evaluation-only, CPU)
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_engine.api import TrainingEngine

# 4 windowallpositive samples: mode=[0,0,1,2]→true_dir=[1,1,0,0]; logits→pred=[1,1,1,0] ⇒ 3/4
m = TrainingEngine._compute_direction_metrics(
    [torch.tensor([[3.0], [3.0], [3.0], [-3.0]])],      # window-level dir_logits [4,1]
    [torch.tensor([0, 0, 1, 2])],                        # window-level hgt_mode [4]
    [torch.tensor([1, 1, 1, 1])])                        # window-level labels [4] (independent of contig aggregation)
assert m["direction_n"] == 4 and m["direction_correct"] == 3, m
assert m["direction_accuracy"] == 0.75 and m["direction_evaluable"] is True and m["direction_status"] == "OK", m
# no-sample path: null / all-negative samples → NOT_APPLICABLE with accuracy=None (0.0 must not stand in for nonexistent)
e = TrainingEngine._compute_direction_metrics([], [], [])
n = TrainingEngine._compute_direction_metrics([torch.tensor([[2.0], [2.0]])], [torch.tensor([0, 1])], [torch.tensor([0, 0])])
assert e["direction_status"] == "NOT_APPLICABLE" and e["direction_accuracy"] is None, e
assert n["direction_status"] == "NOT_APPLICABLE" and n["direction_accuracy"] is None and n["direction_n"] == 0, n
print("PASS: direction_n=4, accuracy=0.75, empty/no-pos -> NOT_APPLICABLE (accuracy=None)")
