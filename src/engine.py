"""帧同步战斗引擎：确定性、整数运算、固定种子RNG。

核心不变量：
- 所有战斗计算用整数，禁止浮点
- RNG用固定种子LCG，确定性
- 每帧输入按顺序应用
- 状态可序列化，回放一致
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Fighter:
    fid: str
    hp: int
    max_hp: int
    attack: int
    defense: int
    crit_rate: int  # 整数百分比 0-100
    crit_dmg: int  # 整数百分比，如150=1.5倍


class FrameSyncBattle:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng_state = seed
        self.frame = 0
        self.fighters: Dict[str, Fighter] = {}
        self.log: List[str] = []

    def _next_rng(self) -> int:
        """LCG随机数，确定性。BUG：没有用固定种子，用了全局random。"""
        # BUG：应该用self.rng_state做LCG，但实际用了Python的random
        import random
        return random.randint(0, 1000000)

    def add_fighter(self, f: Fighter) -> None:
        self.fighters[f.fid] = f

    def _calc_damage(self, attacker: Fighter, defender: Fighter) -> int:
        """计算伤害。BUG：用了浮点数除法。"""
        # BUG：用了浮点数，正确应该是整数运算
        base = attacker.attack * (100 - defender.defense) / 100.0
        # 暴击判定
        roll = self._next_rng() % 100
        if roll < attacker.crit_rate:
            base = base * attacker.crit_dmg / 100.0
        return int(base)

    def tick(self, inputs: List[dict]) -> None:
        """推进一帧。inputs按顺序执行。"""
        self.frame += 1
        for inp in inputs:
            if inp["type"] == "attack":
                atk = self.fighters[inp["from"]]
                tgt = self.fighters[inp["to"]]
                dmg = self._calc_damage(atk, tgt)
                tgt.hp = max(0, tgt.hp - dmg)
                self.log.append(f"frame{self.frame} {atk.fid} hits {tgt.fid} for {dmg}")

    def serialize(self) -> dict:
        return {
            "seed": self.seed,
            "rng_state": self.rng_state,
            "frame": self.frame,
            "fighters": {k: v.__dict__ for k, v in self.fighters.items()},
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FrameSyncBattle":
        b = cls(seed=data["seed"])
        b.rng_state = data["rng_state"]
        b.frame = data["frame"]
        for fid, fd in data["fighters"].items():
            b.fighters[fid] = Fighter(**fd)
        return b
