"""Hand-checked cases. Run: python tests/test_env.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from yard.env import retrieval_cost, relocate_target

# Stack 0 = [2, 0] (rank 0 on top): pulling 0 then 1 then 2 needs no relocation for 0;
# rank 1 sits in stack 1 alone. Total 0.
assert retrieval_cost([[2, 0], [1], []], 3) == 0

# Stack 0 = [0, 1, 2] (0 at the bottom): to pull 0 we must move 2 then 1 -> 2 relocations.
# Blocker 2 goes to an empty-ish stack, blocker 1 likewise; afterwards no more moves.
assert retrieval_cost([[0, 1, 2], [], []], 3) == 2

# Relocation rule: blocker rank 1 from stack 0; stack 1 top is 5 (later), stack 2 top is 3 (later,
# smaller gap), so stack 2 is chosen.
assert relocate_target([[1], [5], [3]], 3, 0, 1) == 2
# If every other top leaves earlier, choose the one that leaves latest (top 4 over top 2).
assert relocate_target([[9], [2], [4]], 3, 0, 8) == 2

# Conservation: relocation never loses containers.
st = [[4, 0, 3], [1, 5], [2]]
retrieval_cost(st, 4)
assert sum(len(s) for s in st) == 0
print("env tests passed")
