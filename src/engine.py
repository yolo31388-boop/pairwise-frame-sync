"""帧同步战斗引擎：确定性、整数运算、固定种子RNG。

核心不变量：
- 所有战斗计算用整数，禁止浮点数参与状态变更（浮点仅用于显示）
- RNG 用固定种子的 LCG，相同种子产生相同序列
- 伤害/治疗/属性修正全部整数运算，除法用整除（向下取整）
- 每帧输入按顺序应用，时间推进基于帧数而非真实时间
- 状态可序列化/反序列化，回放到指定帧与实时战斗完全一致
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# LCG 参数（Numerical Recipes），模 2^32，全周期，跨平台一致
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32


def _require_int(value, name: str) -> int:
    """状态变更只接受整数；浮点数一律拒绝，防止精度误差渗入逻辑。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int，禁止浮点参与状态变更: {value!r}")
    return value


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
        self.seed = _require_int(seed, "seed")
        self.rng_state = self.seed
        self.frame = 0
        self.fighters: Dict[str, Fighter] = {}
        self.log: List[str] = []

    def _next_rng(self) -> int:
        """LCG 随机数：固定种子、纯整数、确定性，相同种子产生相同序列。"""
        self.rng_state = (self.rng_state * _LCG_A + _LCG_C) % _LCG_M
        return self.rng_state

    def add_fighter(self, f: Fighter) -> None:
        self.fighters[f.fid] = f

    def _calc_damage(self, attacker: Fighter, defender: Fighter) -> int:
        """计算伤害：全程整数运算，除法用整除（向下取整），结果不为负。

        base = attack * (100 - defense) // 100
        暴击时 base = base * crit_dmg // 100
        """
        base = attacker.attack * (100 - defender.defense) // 100
        # 暴击判定：每次攻击消耗且仅消耗一个随机数
        roll = self._next_rng() % 100
        if roll < attacker.crit_rate:
            base = base * attacker.crit_dmg // 100
        return max(0, base)

    def tick(self, inputs: List[dict]) -> None:
        """推进一帧。inputs 按列表顺序依次应用，时间只随帧数前进。"""
        self.frame += 1
        for inp in inputs:
            itype = inp["type"]
            if itype == "attack":
                atk = self.fighters[inp["from"]]
                tgt = self.fighters[inp["to"]]
                dmg = self._calc_damage(atk, tgt)
                tgt.hp = max(0, tgt.hp - dmg)
                self.log.append(f"frame{self.frame} {atk.fid} hits {tgt.fid} for {dmg}")
            elif itype == "heal":
                tgt = self.fighters[inp["to"]]
                amount = _require_int(inp["amount"], "heal amount")
                applied = min(amount, tgt.max_hp - tgt.hp)
                applied = max(0, applied)
                tgt.hp += applied
                self.log.append(f"frame{self.frame} {tgt.fid} healed for {applied}")
            elif itype == "modify":
                tgt = self.fighters[inp["to"]]
                attr = inp["attr"]
                percent = _require_int(inp["percent"], "modify percent")
                old = _require_int(getattr(tgt, attr), f"fighter.{attr}")
                new = max(0, old * percent // 100)
                setattr(tgt, attr, new)
                self.log.append(f"frame{self.frame} {tgt.fid} {attr} {old}->{new}")
            else:
                raise ValueError(f"未知输入类型: {itype!r}")

    def hp_ratio(self, fid: str) -> float:
        """仅供 UI 显示的浮点值，不参与任何状态变更。"""
        f = self.fighters[fid]
        return f.hp / f.max_hp

    def serialize(self) -> dict:
        return {
            "seed": self.seed,
            "rng_state": self.rng_state,
            "frame": self.frame,
            "fighters": {k: dict(v.__dict__) for k, v in self.fighters.items()},
            "log": list(self.log),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "FrameSyncBattle":
        b = cls(seed=data["seed"])
        b.rng_state = data["rng_state"]
        b.frame = data["frame"]
        for fid, fd in data["fighters"].items():
            b.fighters[fid] = Fighter(**fd)
        b.log = list(data.get("log", []))
        return b
