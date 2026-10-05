"""
游戏内帧同步与确定性渲染系统 - 核心模块

纯标准库实现，保证同一组输入在任何客户端上产生完全一致的结果：

- 帧同步 (frame_sync):      固定步长帧锁定 + 锁步输入同步
- 确定性渲染 (deterministic_rendering): 确定性随机数 + 确定性排序
- 确定性物理 (deterministic_physics):   确定性积分 + 确定性碰撞
- 确定性输入 (deterministic_input):     输入采样 + 输入回放
- 确定性随机数 (deterministic_random):  种子初始化 + 确定性序列
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    frame_rate: float = 30.0
    seed: int = 0
    gravity: float = 0.0


def _ordered_keys(mapping):
    """确定性键序：同类型按键排序，混合类型退化为字符串排序。"""
    try:
        return sorted(mapping)
    except TypeError:
        return sorted(mapping, key=str)


class DeterministicRandom:
    """确定性随机数生成器（splitmix64），同一种子产生同一序列。"""

    MASK64 = (1 << 64) - 1
    _GAMMA = 0x9E3779B97F4A7C15

    def __init__(self, seed: int = 0):
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self._seed = int(seed) & self.MASK64
        self._state = (self._seed ^ self._GAMMA) & self.MASK64
        if self._state == 0:
            self._state = self._GAMMA
        self._count = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def count(self) -> int:
        return self._count

    def _next_u64(self) -> int:
        self._state = (self._state + self._GAMMA) & self.MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK64
        z = z ^ (z >> 31)
        self._count += 1
        return z

    def random(self) -> float:
        """[0.0, 1.0) 均匀分布。"""
        return (self._next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("empty range for randint")
        return a + self._next_u64() % (b - a + 1)

    def randrange(self, start, stop=None, step=1) -> int:
        if stop is None:
            start, stop = 0, start
        n = len(range(start, stop, step))
        if n <= 0:
            raise ValueError("empty range for randrange")
        return start + step * (self._next_u64() % n)

    def choice(self, seq):
        if not seq:
            raise IndexError("cannot choose from an empty sequence")
        return seq[self._next_u64() % len(seq)]

    def shuffle(self, seq):
        for i in range(len(seq) - 1, 0, -1):
            j = self._next_u64() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    def sequence(self, n: int) -> List[float]:
        """确定性随机序列：同一种子生成完全相同的序列。"""
        return [self.random() for _ in range(int(n))]

    def snapshot(self) -> Dict:
        return {"seed": self._seed, "state": self._state, "count": self._count}

    def restore(self, snap: Dict):
        self._seed = int(snap["seed"]) & self.MASK64
        self._state = int(snap["state"]) & self.MASK64
        self._count = int(snap["count"])


class FrameSync:
    """帧同步与确定性模拟核心。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.frame_rate = float(self.config.get("frame_rate", 30.0))
        self.frame_dt = 1.0 / self.frame_rate
        self.frame = 0
        self._accumulator = 0.0
        self._pending_inputs: Dict[int, Dict[str, object]] = {}
        self._history: List[Dict] = []
        self._state: Dict = {}
        self._rng = DeterministicRandom(self.config.get("seed", 0))
        self._gravity = float(self.config.get("gravity", 0.0))
        self._bodies: Dict = {}
        self._initial_bodies: Dict = {}
        self._next_body_id = 0

    # ------------------------------------------------------------------
    # 帧同步：帧锁定
    # ------------------------------------------------------------------
    def update(self, dt: float = 0.0) -> int:
        """按固定步长推进模拟（帧锁定）。

        真实时间 ``dt`` 进入累加器，每攒满一个 ``frame_dt`` 且该帧输入
        已就绪时推进一步，返回本帧推进的步数。输入未就绪时停帧等待，
        保证所有客户端推进完全相同的帧序列。
        """
        steps = 0
        if dt and dt > 0:
            self._accumulator += float(dt)
            while self._accumulator >= self.frame_dt:
                if not self.inputs_ready():
                    break
                self._accumulator -= self.frame_dt
                self.step()
                steps += 1
        return steps

    def inputs_ready(self, frame: Optional[int] = None) -> bool:
        frame = self.frame if frame is None else int(frame)
        return frame in self._pending_inputs

    # ------------------------------------------------------------------
    # 帧同步：输入同步（锁步）
    # ------------------------------------------------------------------
    def submit_input(self, player_id, data, frame: Optional[int] = None):
        """提交某玩家在某帧的输入，未指定帧时为当前帧。"""
        frame = self.frame if frame is None else int(frame)
        slot = self._pending_inputs.setdefault(frame, {})
        slot[str(player_id)] = self._normalize_input(data)

    def step(self, inputs: Optional[Dict] = None) -> Dict:
        """推进一个逻辑帧：收集输入 -> 确定性模拟 -> 记录历史。"""
        if inputs is None:
            inputs = self._pending_inputs.pop(self.frame, {})
        inputs = {str(k): v for k, v in
                  sorted(inputs.items(), key=lambda kv: str(kv[0]))}
        self._history.append({"frame": self.frame, "inputs": inputs})
        self._simulate(inputs, self.frame_dt)
        self.frame += 1
        return inputs

    def frame_sync(self, inputs: Optional[Dict] = None,
                   dt: Optional[float] = None) -> int:
        """帧同步入口：提交输入并按帧锁定推进，返回推进的帧数。"""
        if inputs:
            for player_id, data in inputs.items():
                self.submit_input(player_id, data)
        if dt is not None:
            return self.update(dt)
        if self.inputs_ready():
            self.step()
            return 1
        return 0

    def _simulate(self, inputs: Dict, dt: float):
        self.physics_step(dt)

    # ------------------------------------------------------------------
    # 确定性随机数：种子 + 序列
    # ------------------------------------------------------------------
    def deterministic_random(self, seed: Optional[int] = None) -> float:
        """确定性随机数；给定种子时先重置序列。同一种子结果一致。"""
        if seed is not None:
            self._rng.set_seed(seed)
        return self._rng.random()

    def random_sequence(self, n: int, seed: Optional[int] = None) -> List[float]:
        """确定性随机序列；同一种子生成完全相同的序列。"""
        if seed is not None:
            self._rng.set_seed(seed)
        return self._rng.sequence(n)

    # ------------------------------------------------------------------
    # 确定性渲染：确定性随机数 + 确定性排序
    # ------------------------------------------------------------------
    def deterministic_rendering(self, objects: Optional[List] = None,
                                rng: Optional[DeterministicRandom] = None) -> Dict:
        """确定性渲染：对象按 (layer, id) 排序，抖动来自确定性随机数。

        无论输入对象顺序如何，输出顺序与随机抖动完全一致。
        """
        rng = rng or self._rng
        objs = list(objects) if objects else []
        ordered = sorted(objs, key=lambda o: (self._layer_of(o), self._sort_key(o)))
        return {"frame": self.frame, "objects": ordered, "jitter": rng.random()}

    @staticmethod
    def _layer_of(obj):
        if isinstance(obj, dict):
            return obj.get("layer", obj.get("z", 0))
        return getattr(obj, "layer", 0)

    @staticmethod
    def _sort_key(obj) -> str:
        if isinstance(obj, dict):
            for key in ("id", "name"):
                if key in obj:
                    return str(obj[key])
            return repr(sorted(obj.items(), key=lambda kv: str(kv[0])))
        return str(obj)

    # ------------------------------------------------------------------
    # 确定性物理：确定性积分 + 确定性碰撞
    # ------------------------------------------------------------------
    def add_body(self, x=0.0, y=0.0, vx=0.0, vy=0.0,
                 radius=1.0, mass=1.0, body_id=None):
        """注册一个物理刚体，返回其 id。"""
        if body_id is None:
            body_id = self._next_body_id
            self._next_body_id += 1
        body = {
            "id": body_id,
            "x": float(x), "y": float(y),
            "vx": float(vx), "vy": float(vy),
            "radius": float(radius), "mass": float(mass),
        }
        self._bodies[body_id] = body
        self._initial_bodies[body_id] = dict(body)
        return body_id

    def deterministic_physics(self, bodies=None, dt: Optional[float] = None):
        """确定性物理：固定顺序积分 + 排序后的成对碰撞求解。

        不指定刚体时推进内部世界并返回快照；指定刚体列表时对该列表
        模拟一步并返回按 id 排序的结果，与输入顺序无关。
        """
        dt = self.frame_dt if dt is None else float(dt)
        if bodies is None:
            self._integrate_all(self._bodies, dt)
            self._resolve_collisions(self._bodies)
            return self.snapshot()
        local = [self._normalize_body(b, i) for i, b in enumerate(bodies)]
        world = {b["id"]: b for b in local}
        self._integrate_all(world, dt)
        self._resolve_collisions(world)
        return [world[k] for k in _ordered_keys(world)]

    def physics_step(self, dt: Optional[float] = None) -> Dict:
        """对内部世界推进一步确定性物理，返回快照。"""
        dt = self.frame_dt if dt is None else float(dt)
        self._integrate_all(self._bodies, dt)
        self._resolve_collisions(self._bodies)
        return self.snapshot()

    def _integrate_all(self, bodies: Dict, dt: float):
        """确定性积分：固定键序的半隐式欧拉积分。"""
        for key in _ordered_keys(bodies):
            b = bodies[key]
            b["vy"] += self._gravity * dt
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt

    def _resolve_collisions(self, bodies: Dict):
        """确定性碰撞：按键序生成有序刚体对，结果与插入顺序无关。"""
        keys = _ordered_keys(bodies)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = bodies[keys[i]], bodies[keys[j]]
                self._resolve_pair(a, b)

    @staticmethod
    def _resolve_pair(a: Dict, b: Dict):
        ra, rb = a.get("radius", 0.0), b.get("radius", 0.0)
        if ra <= 0.0 or rb <= 0.0:
            return
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        dist = math.hypot(dx, dy)
        min_dist = ra + rb
        if dist >= min_dist or dist == 0.0:
            return
        nx, ny = dx / dist, dy / dist
        ma, mb = a.get("mass", 1.0), b.get("mass", 1.0)
        inv_a = 1.0 / ma if ma > 0 else 0.0
        inv_b = 1.0 / mb if mb > 0 else 0.0
        inv_sum = inv_a + inv_b
        if inv_sum == 0.0:
            return
        overlap = min_dist - dist
        a["x"] -= nx * overlap * (inv_a / inv_sum)
        a["y"] -= ny * overlap * (inv_a / inv_sum)
        b["x"] += nx * overlap * (inv_b / inv_sum)
        b["y"] += ny * overlap * (inv_b / inv_sum)
        rvx, rvy = b["vx"] - a["vx"], b["vy"] - a["vy"]
        vn = rvx * nx + rvy * ny
        if vn < 0.0:
            impulse = -2.0 * vn / inv_sum
            a["vx"] -= impulse * inv_a * nx
            a["vy"] -= impulse * inv_a * ny
            b["vx"] += impulse * inv_b * nx
            b["vy"] += impulse * inv_b * ny

    @staticmethod
    def _normalize_body(body, index) -> Dict:
        return {
            "id": body.get("id", index),
            "x": float(body.get("x", 0.0)), "y": float(body.get("y", 0.0)),
            "vx": float(body.get("vx", 0.0)), "vy": float(body.get("vy", 0.0)),
            "radius": float(body.get("radius", 1.0)),
            "mass": float(body.get("mass", 1.0)),
        }

    # ------------------------------------------------------------------
    # 确定性输入：输入采样 + 输入回放
    # ------------------------------------------------------------------
    def deterministic_input(self, player_id=None, data=None, frame: Optional[int] = None):
        """确定性输入采样。

        给定 ``data`` 时规范化并提交该输入（键排序、id 字符串化），
        返回规范化结果；否则返回最近一次采样到的该玩家输入。
        """
        if data is not None:
            normalized = self._normalize_input(data)
            self.submit_input(player_id, normalized, frame)
            return normalized
        if player_id is None:
            return self._history[-1]["inputs"] if self._history else {}
        pid = str(player_id)
        for record in reversed(self._history):
            if pid in record["inputs"]:
                return record["inputs"][pid]
        return None

    @staticmethod
    def _normalize_input(data):
        if data is None:
            return {}
        if isinstance(data, dict):
            return {str(k): data[k] for k in sorted(data, key=lambda k: str(k))}
        return data

    def input_history(self) -> List[Dict]:
        """按帧返回历史输入序列，可用于回放。"""
        return [record["inputs"] for record in self._history]

    def replay_inputs(self, history: Optional[List] = None) -> List[Dict]:
        """输入回放：重放输入序列，精确复现模拟过程。

        不指定序列时回放缓冲区中的历史：先恢复到初始状态再逐帧重放，
        回放后的世界状态与回放前完全一致。
        """
        if history is None:
            history = list(self._history)
            self._restore_initial()
        results = []
        for record in history:
            if isinstance(record, dict) and "inputs" in record:
                inputs = record["inputs"]
            else:
                inputs = record
            results.append(self.step(inputs))
        return results

    def _restore_initial(self):
        self.frame = 0
        self._accumulator = 0.0
        self._pending_inputs = {}
        self._history = []
        self._bodies = {k: dict(v) for k, v in self._initial_bodies.items()}
        self._rng.set_seed(self.config.get("seed", 0))

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict:
        """确定性世界快照，可用于跨客户端一致性校验。"""
        return {
            "frame": self.frame,
            "bodies": {k: dict(self._bodies[k]) for k in _ordered_keys(self._bodies)},
            "rng": self._rng.snapshot(),
        }

    def reset(self):
        self.frame = 0
        self._accumulator = 0.0
        self._pending_inputs = {}
        self._history = []
        self._state = {}
        self._bodies = {}
        self._initial_bodies = {}
        self._next_body_id = 0
        self._rng.set_seed(self.config.get("seed", 0))

    # 常用别名
    render = deterministic_rendering
    physics = deterministic_physics
    sample_input = deterministic_input
    replay = replay_inputs
