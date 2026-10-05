"""
游戏内帧同步与确定性渲染系统 - 核心模块

实现内容：
- frame_sync()          帧同步：帧锁定（固定步长）+ 输入同步（lockstep）
- deterministic_rendering() 确定性渲染：确定性随机数 + 确定性排序
- deterministic_physics()   确定性物理：确定性积分 + 确定性碰撞
- deterministic_input()     确定性输入：输入采样 + 输入回放
- deterministic_random()    确定性随机数：种子 + 序列
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time

DEFAULT_FRAME_RATE = 60
DEFAULT_SEED = 0
DEFAULT_PLAYER_IDS = ("p1", "p2")
DEFAULT_PLAYER_SPACING = 10.0


@dataclass
class Config:
    frame_rate: int = DEFAULT_FRAME_RATE
    seed: int = DEFAULT_SEED
    player_ids: Tuple[str, ...] = DEFAULT_PLAYER_IDS


class DeterministicRandom:
    """确定性随机数生成器（LCG，32 位，与平台无关）。

    - 种子：seed() 可用任意整数种子初始化/重置序列
    - 序列：相同种子产生完全相同的随机序列
    """

    _MODULUS = 1 << 32
    _MULTIPLIER = 1664525
    _INCREMENT = 1013904223

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed(seed)

    def seed(self, seed: int):
        self._seed = int(seed) % self._MODULUS
        self._state = self._seed

    def next_u32(self) -> int:
        self._state = (self._MULTIPLIER * self._state + self._INCREMENT) % self._MODULUS
        return self._state

    def random(self) -> float:
        """返回 [0.0, 1.0) 区间内的确定性浮点数。"""
        return self.next_u32() / self._MODULUS

    def randint(self, low: int, high: int) -> int:
        """返回 [low, high] 闭区间内的确定性整数。"""
        if high < low:
            raise ValueError("high 必须大于等于 low")
        span = high - low + 1
        return low + self.next_u32() % span

    def sequence(self, count: int) -> List[float]:
        """生成确定性的随机数序列。"""
        return [self.random() for _ in range(count)]


def _mix_seed(seed: int, frame: int = 0) -> int:
    """把 (种子, 帧号) 混合为单一 32 位确定性种子。"""
    return (int(seed) * 2654435761 + int(frame)) % (1 << 32)


def deterministic_random(seed: int = DEFAULT_SEED, count: int = 0):
    """确定性随机数入口：种子 + 序列。

    count > 0 时返回长度为 count 的确定性随机序列，否则返回生成器实例。
    """
    rng = DeterministicRandom(seed)
    if count > 0:
        return rng.sequence(count)
    return rng


def _canonical_input(raw: Optional[Dict]) -> Dict:
    """把原始输入规整为确定性快照：固定字段、定点化（4 位小数）。"""
    raw = raw or {}
    return {
        "x": round(float(raw.get("x", 0.0)), 4),
        "y": round(float(raw.get("y", 0.0)), 4),
        "buttons": tuple(sorted(str(b) for b in raw.get("buttons", ()))),
    }


def deterministic_input(inputs: Optional[Dict] = None, replay: bool = False,
                        input_log: Optional[List[Dict]] = None):
    """确定性输入入口：输入采样 + 输入回放。

    - 采样：replay=False，把原始输入规整为确定性快照
    - 回放：replay=True，从 input_log 按帧序回放输入快照
    """
    if replay:
        log = input_log if input_log is not None else []
        return [dict(snapshot) for snapshot in log]
    return {pid: _canonical_input(raw) for pid, raw in (inputs or {}).items()}


def deterministic_physics(bodies: List[Dict], dt: float,
                          rng: Optional[DeterministicRandom] = None) -> List[Dict]:
    """确定性物理：确定性积分 + 确定性碰撞。

    - 积分：固定步长 dt 的半隐式欧拉积分，所有客户端结果一致
    - 碰撞：等质量完全弹性碰撞，按稳定顺序逐对处理，结果与输入顺序无关
    """
    if rng is None:
        rng = DeterministicRandom(DEFAULT_SEED)
    dt = float(dt)
    result = []
    for index, body in enumerate(bodies):
        new_body = {
            "id": str(body.get("id", "body%d" % index)),
            "x": float(body.get("x", 0.0)),
            "y": float(body.get("y", 0.0)),
            "vx": float(body.get("vx", 0.0)),
            "vy": float(body.get("vy", 0.0)),
            "radius": float(body.get("radius", 0.5)),
        }
        ax = float(body.get("ax", 0.0))
        ay = float(body.get("ay", 0.0))
        # 确定性积分：半隐式欧拉（先速度后位置）
        new_body["vx"] += ax * dt
        new_body["vy"] += ay * dt
        new_body["x"] += new_body["vx"] * dt
        new_body["y"] += new_body["vy"] * dt
        result.append(new_body)

    # 确定性碰撞：按 (id, id) 稳定顺序处理所有碰撞对
    ordered = sorted(range(len(result)), key=lambda i: result[i]["id"])
    for left_pos in range(len(ordered)):
        for right_pos in range(left_pos + 1, len(ordered)):
            i, j = ordered[left_pos], ordered[right_pos]
            a, b = result[i], result[j]
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            min_dist = a["radius"] + b["radius"]
            dist_sq = dx * dx + dy * dy
            if dist_sq >= min_dist * min_dist:
                continue
            dist = math.sqrt(dist_sq)
            if dist == 0.0:
                # 完全重叠：用确定性随机数选择分离方向
                angle = rng.random() * 2.0 * math.pi
                nx, ny = math.cos(angle), math.sin(angle)
                dist = 1e-9
            else:
                nx, ny = dx / dist, dy / dist
            # 位置分离（各退一半穿透量）
            overlap = min_dist - dist
            a["x"] -= nx * overlap * 0.5
            a["y"] -= ny * overlap * 0.5
            b["x"] += nx * overlap * 0.5
            b["y"] += ny * overlap * 0.5
            # 等质量完全弹性碰撞：交换法向速度分量
            rel_vn = (b["vx"] - a["vx"]) * nx + (b["vy"] - a["vy"]) * ny
            if rel_vn < 0.0:
                a["vx"] += rel_vn * nx
                a["vy"] += rel_vn * ny
                b["vx"] -= rel_vn * nx
                b["vy"] -= rel_vn * ny
    # 确定性排序：输出按 id 稳定排列，与输入列表顺序无关
    return sorted(result, key=lambda body: body["id"])


def deterministic_rendering(objects: List[Dict], frame: int = 0,
                            seed: int = DEFAULT_SEED) -> List[Dict]:
    """确定性渲染：确定性随机数 + 确定性排序。

    - 排序：按 (z_index, id) 稳定排序，任意输入顺序得到相同渲染顺序
    - 随机数：每帧由 (seed, frame) 派生确定性随机流，每个对象再按
      排序后的索引派生子流，渲染结果跨客户端完全一致
    """
    frame_rng = DeterministicRandom(_mix_seed(seed, frame))
    ordered = sorted(
        objects,
        key=lambda obj: (int(obj.get("z_index", 0)), str(obj.get("id", ""))),
    )
    rendered = []
    for index, obj in enumerate(ordered):
        obj_rng = DeterministicRandom((frame_rng.next_u32() + index * 40503) % (1 << 32))
        rendered.append({
            "id": str(obj.get("id", "")),
            "z_index": int(obj.get("z_index", 0)),
            "order": index,
            "jitter": obj_rng.random(),
        })
    return rendered


class FrameSync:
    """帧同步核心：帧锁定 + 输入同步（lockstep）。

    - 帧锁定：固定步长 1/frame_rate，update(dt) 累积时间并整帧推进
    - 输入同步：所有客户端按帧收集相同的玩家输入后再模拟，
      相同种子 + 相同输入序列 => 所有客户端状态逐位一致
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self.frame_rate = int(self.config.get("frame_rate", DEFAULT_FRAME_RATE))
        self.fixed_dt = 1.0 / self.frame_rate
        self.seed = int(self.config.get("seed", DEFAULT_SEED))
        self.player_ids = tuple(self.config.get("player_ids", DEFAULT_PLAYER_IDS))
        self.player_spacing = float(self.config.get("player_spacing", DEFAULT_PLAYER_SPACING))
        self.reset()

    def reset(self):
        self._state = {
            "frame": 0,
            "players": {
                pid: {
                    "id": pid,
                    "x": index * self.player_spacing,
                    "y": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "radius": 0.5,
                }
                for index, pid in enumerate(self.player_ids)
            },
        }
        self._history = []
        self._input_log = []
        self._pending_inputs = {}
        self._accumulator = 0.0
        self.rng = DeterministicRandom(self.seed)

    # ---- 输入同步 ----

    def submit_input(self, player_id: str, raw_input: Optional[Dict],
                     frame: Optional[int] = None):
        """提交某玩家在某帧的输入（默认当前帧），存入输入缓冲区。"""
        target_frame = self._state["frame"] if frame is None else int(frame)
        self._pending_inputs.setdefault(target_frame, {})[player_id] = _canonical_input(raw_input)

    def deterministic_input(self, inputs: Optional[Dict] = None, replay: bool = False):
        """确定性输入：输入采样 + 输入回放。"""
        if replay:
            return deterministic_input(replay=True, input_log=self._input_log)
        sampled = deterministic_input(inputs)
        frame_inputs = self._pending_inputs.setdefault(self._state["frame"], {})
        frame_inputs.update(sampled)
        return sampled

    # ---- 帧同步 ----

    def frame_sync(self, inputs: Optional[Dict] = None) -> Dict:
        """按 lockstep 推进一帧：收集输入 -> 采样 -> 记录 -> 确定性模拟。"""
        frame = self._state["frame"]
        if inputs:
            self.deterministic_input(inputs)
        frame_inputs = self._pending_inputs.pop(frame, {})
        snapshot = {pid: frame_inputs.get(pid, _canonical_input(None)) for pid in self.player_ids}
        self._input_log.append({"frame": frame, "inputs": dict(snapshot)})

        bodies = []
        for pid in self.player_ids:
            body = dict(self._state["players"][pid])
            move = snapshot[pid]
            body["ax"] = move["x"]
            body["ay"] = move["y"]
            bodies.append(body)
        stepped = deterministic_physics(bodies, self.fixed_dt, self.rng)
        for body in stepped:
            body.pop("ax", None)
            body.pop("ay", None)
            self._state["players"][body["id"]] = body
        self._state["frame"] = frame + 1
        self._history.append({pid: dict(body) for pid, body in self._state["players"].items()})
        return self._state

    def update(self, dt: float):
        """帧锁定更新：按固定步长整帧推进，返回本帧锁定的帧数。"""
        if dt <= 0.0:
            return 0
        self._accumulator += float(dt)
        steps = 0
        while self._accumulator + 1e-12 >= self.fixed_dt:
            self._accumulator -= self.fixed_dt
            self.frame_sync()
            steps += 1
        return steps

    # ---- 确定性渲染 ----

    def deterministic_rendering(self, objects: Optional[List[Dict]] = None) -> List[Dict]:
        """对当前帧的对象做确定性渲染（确定性排序 + 确定性随机数）。"""
        if objects is None:
            objects = list(self._state["players"].values())
        return deterministic_rendering(objects, frame=self._state["frame"], seed=self.seed)

    # ---- 确定性物理 ----

    def deterministic_physics(self, bodies: List[Dict], dt: Optional[float] = None) -> List[Dict]:
        """确定性物理步进（确定性积分 + 确定性碰撞）。"""
        step = self.fixed_dt if dt is None else float(dt)
        return deterministic_physics(bodies, step, self.rng)

    # ---- 确定性随机数 ----

    def deterministic_random(self, count: int = 0):
        """确定性随机数：count > 0 返回序列，否则返回下一个随机数。"""
        if count > 0:
            return self.rng.sequence(count)
        return self.rng.random()
