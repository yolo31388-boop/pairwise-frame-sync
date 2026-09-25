"""帧同步战斗引擎：确定性、整数运算、固定种子RNG。

核心不变量：
- 所有战斗计算用整数，禁止浮点参与状态变更（浮点仅用于显示）
- RNG用固定种子LCG，相同种子产生相同序列
- 除法一律整除（向下取整），舍入规则明确
- 每帧输入按顺序应用
- 状态可序列化/反序列化，序列化后继续战斗结果一致
- 时间推进基于帧数，与真实时间无关
- 回放到指定帧的状态与实时战斗到该帧完全一致
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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
    # LCG参数（glibc），全平台整数运算，结果与平台无关
    LCG_A = 1103515245
    LCG_C = 12345
    LCG_M = 2 ** 31

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng_state = seed % self.LCG_M
        self.frame = 0
        self.fighters: Dict[str, Fighter] = {}
        self.log: List[str] = []
        self.inputs_log: List[List[dict]] = []  # 每帧输入记录，用于回放
        self._initial_fighters: Dict[str, Fighter] = {}

    def _next_rng(self) -> int:
        """固定种子LCG，确定性推进内部状态。"""
        self.rng_state = (self.rng_state * self.LCG_A + self.LCG_C) % self.LCG_M
        return self.rng_state

    def add_fighter(self, f: Fighter) -> None:
        self.fighters[f.fid] = f
        if f.fid not in self._initial_fighters:
            self._initial_fighters[f.fid] = Fighter(**f.__dict__)

    def _calc_damage(self, attacker: Fighter, defender: Fighter) -> int:
        """计算伤害。纯整数运算，除法用整除（向下取整）。"""
        base = attacker.attack * (100 - defender.defense) // 100
        roll = self._next_rng() % 100
        if roll < attacker.crit_rate:
            base = base * attacker.crit_dmg // 100
        return base

    def _calc_heal(self, amount: int) -> int:
        """治疗量必须是整数，直接取整数值。"""
        return int(amount)

    def tick(self, inputs: List[dict]) -> None:
        """推进一帧。inputs严格按顺序执行。时间只随帧数推进。"""
        self.frame += 1
        self.inputs_log.append([dict(i) for i in inputs])
        for inp in inputs:
            kind = inp["type"]
            if kind == "attack":
                atk = self.fighters[inp["from"]]
                tgt = self.fighters[inp["to"]]
                dmg = self._calc_damage(atk, tgt)
                tgt.hp = max(0, tgt.hp - dmg)
                self.log.append(f"frame{self.frame} {atk.fid} hits {tgt.fid} for {dmg}")
            elif kind == "heal":
                tgt = self.fighters[inp["to"]]
                amount = self._calc_heal(inp["amount"])
                before = tgt.hp
                tgt.hp = min(tgt.max_hp, tgt.hp + amount)
                self.log.append(
                    f"frame{self.frame} {tgt.fid} healed for {tgt.hp - before}"
                )
            elif kind == "modify":
                tgt = self.fighters[inp["to"]]
                attr = inp["attr"]
                old = getattr(tgt, attr)
                if "pct" in inp:
                    # 百分比修正：整数乘法后整除（向下取整）
                    new = old * (100 + inp["pct"]) // 100
                else:
                    new = old + inp["delta"]
                setattr(tgt, attr, new)
                self.log.append(
                    f"frame{self.frame} {tgt.fid} {attr} {old}->{new}"
                )

    def display_hp_percent(self, fid: str) -> float:
        """浮点仅用于显示，不参与任何状态变更。"""
        f = self.fighters[fid]
        return f.hp * 100.0 / f.max_hp

    def replay_to(self, frame: int) -> "FrameSyncBattle":
        """从初始状态回放到指定帧，结果必须与实时战斗到该帧一致。"""
        b = FrameSyncBattle(seed=self.seed)
        for f in self._initial_fighters.values():
            b.add_fighter(Fighter(**f.__dict__))
        for inputs in self.inputs_log[:frame]:
            b.tick(inputs)
        return b

    def serialize(self) -> dict:
        return {
            "seed": self.seed,
            "rng_state": self.rng_state,
            "frame": self.frame,
            "fighters": {k: v.__dict__ for k, v in self.fighters.items()},
            "log": list(self.log),
            "inputs_log": [[dict(i) for i in frame] for frame in self.inputs_log],
            "initial_fighters": {
                k: v.__dict__ for k, v in self._initial_fighters.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FrameSyncBattle":
        b = cls(seed=data["seed"])
        b.rng_state = data["rng_state"]
        b.frame = data["frame"]
        for fid, fd in data["fighters"].items():
            b.fighters[fid] = Fighter(**fd)
        b.log = list(data.get("log", []))
        b.inputs_log = [[dict(i) for i in frame] for frame in data.get("inputs_log", [])]
        for fid, fd in data.get("initial_fighters", {}).items():
            b._initial_fighters[fid] = Fighter(**fd)
        return b
