"""
游戏内帧同步与确定性渲染系统 - 验收测试
运行方式：python -m pytest tests/test_frame_sync.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.frame_sync import (
    FrameSync,
    DeterministicRandom,
    deterministic_random,
    deterministic_input,
    deterministic_physics,
    deterministic_rendering,
)


def run_frames(system, frames, inputs_per_frame=None):
    for frame in range(frames):
        inputs = (inputs_per_frame or {}).get(frame)
        system.frame_sync(inputs)


class TestFrameSync:
    def setup_method(self):
        self.system = FrameSync()

    # ---- 确定性随机数：种子 + 序列 ----

    def test_case_01(self):
        """确定性随机数：相同种子产生相同序列"""
        seq_a = deterministic_random(seed=42, count=10)
        seq_b = deterministic_random(seed=42, count=10)
        assert seq_a == seq_b
        assert len(seq_a) == 10
        assert all(0.0 <= value < 1.0 for value in seq_a)

    def test_case_02(self):
        """确定性随机数：不同种子产生不同序列，且可重复播种重置"""
        rng = DeterministicRandom(1)
        first = rng.sequence(5)
        rng.seed(2)
        other = rng.sequence(5)
        assert first != other
        rng.seed(1)
        assert rng.sequence(5) == first

    def test_case_03(self):
        """确定性随机数：跨实例/跨客户端序列逐位一致"""
        rng_a = DeterministicRandom(7)
        rng_b = DeterministicRandom(7)
        assert [rng_a.random() for _ in range(100)] == [rng_b.random() for _ in range(100)]
        assert rng_a.randint(1, 6) == rng_b.randint(1, 6)

    # ---- 帧同步：帧锁定 + 输入同步 ----

    def test_case_04(self):
        """帧锁定：固定步长推进，不足一帧不模拟"""
        system = FrameSync({"frame_rate": 60})
        assert system.fixed_dt == pytest.approx(1.0 / 60.0)
        assert system.update(1.0 / 120.0) == 0
        assert system._state["frame"] == 0
        assert system.update(1.0 / 60.0) == 1
        assert system._state["frame"] == 1
        assert system.update(1.0) == 60
        assert system._state["frame"] == 61

    def test_case_05(self):
        """帧锁定：相同输入序列下两个客户端状态逐位一致"""
        client_a = FrameSync({"seed": 99})
        client_b = FrameSync({"seed": 99})
        inputs = {frame: {"p1": {"x": 1.0}, "p2": {"x": -1.0}} for frame in range(30)}
        run_frames(client_a, 30, inputs)
        run_frames(client_b, 30, inputs)
        assert client_a._state == client_b._state
        assert client_a._input_log == client_b._input_log

    def test_case_06(self):
        """帧锁定：重置后相同操作重现相同状态"""
        system = FrameSync({"seed": 5})
        run_frames(system, 10, {0: {"p1": {"x": 0.5}}})
        first = dict(system._state)
        system.reset()
        assert system._state["frame"] == 0
        run_frames(system, 10, {0: {"p1": {"x": 0.5}}})
        assert system._state == first

    def test_case_07(self):
        """输入同步：lockstep 下输入驱动确定的状态演化"""
        system = FrameSync()
        system.frame_sync({"p1": {"x": 1.0, "y": 0.0}, "p2": {"x": 0.0, "y": 1.0}})
        players = system._state["players"]
        dt = system.fixed_dt
        assert players["p1"]["x"] == pytest.approx(0.0 + 1.0 * dt * dt)
        assert players["p1"]["vx"] == pytest.approx(1.0 * dt)
        assert players["p2"]["y"] == pytest.approx(1.0 * dt * dt)
        assert players["p2"]["vy"] == pytest.approx(1.0 * dt)

    def test_case_08(self):
        """输入同步：submit_input 提交的输入在对应帧生效"""
        system = FrameSync()
        system.submit_input("p1", {"x": 2.0})
        system.frame_sync()
        assert system._input_log[0]["inputs"]["p1"]["x"] == pytest.approx(2.0)
        assert system._state["players"]["p1"]["vx"] == pytest.approx(2.0 * system.fixed_dt)

    # ---- 确定性渲染：确定性随机数 + 确定性排序 ----

    def test_case_09(self):
        """确定性渲染：相同对象与帧号渲染结果完全一致"""
        objects = [
            {"id": "hero", "z_index": 1},
            {"id": "bg", "z_index": 0},
            {"id": "fx", "z_index": 1},
        ]
        first = deterministic_rendering(objects, frame=10, seed=3)
        second = deterministic_rendering(list(reversed(objects)), frame=10, seed=3)
        assert first == second
        assert [item["id"] for item in first] == ["bg", "fx", "hero"]
        assert [item["order"] for item in first] == [0, 1, 2]

    def test_case_10(self):
        """确定性渲染：渲染抖动来自确定性随机数，帧号不同则不同"""
        objects = [{"id": "a"}, {"id": "b"}]
        frame_one = deterministic_rendering(objects, frame=1, seed=8)
        frame_one_again = deterministic_rendering(objects, frame=1, seed=8)
        frame_two = deterministic_rendering(objects, frame=2, seed=8)
        assert frame_one == frame_one_again
        assert [item["jitter"] for item in frame_one] != [item["jitter"] for item in frame_two]
        assert all(0.0 <= item["jitter"] < 1.0 for item in frame_one)

    # ---- 确定性物理：确定性积分 + 确定性碰撞 ----

    def test_case_11(self):
        """确定性物理：固定步长积分结果确定且与顺序无关"""
        bodies = [
            {"id": "a", "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "ax": 0.5, "ay": 0.0},
            {"id": "b", "x": 5.0, "y": 0.0, "vx": -1.0, "vy": 0.0},
        ]
        dt = 1.0 / 60.0
        first = deterministic_physics(bodies, dt)
        second = deterministic_physics(list(reversed(bodies)), dt)
        assert first == second
        body_a = next(b for b in first if b["id"] == "a")
        assert body_a["vx"] == pytest.approx(1.0 + 0.5 * dt)
        assert body_a["x"] == pytest.approx((1.0 + 0.5 * dt) * dt)

    def test_case_12(self):
        """确定性物理：碰撞处理确定且守恒（等质量弹性碰撞交换法向速度）"""
        bodies = [
            {"id": "a", "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "radius": 0.5},
            {"id": "b", "x": 0.9, "y": 0.0, "vx": -1.0, "vy": 0.0, "radius": 0.5},
        ]
        dt = 1.0 / 60.0
        first = deterministic_physics(bodies, dt)
        second = deterministic_physics(list(reversed(bodies)), dt)
        assert first == second
        body_a = next(b for b in first if b["id"] == "a")
        body_b = next(b for b in first if b["id"] == "b")
        assert body_a["vx"] == pytest.approx(-1.0)
        assert body_b["vx"] == pytest.approx(1.0)
        distance = abs(body_b["x"] - body_a["x"])
        assert distance >= 1.0 - 1e-9

    # ---- 确定性输入：输入采样 + 输入回放 ----

    def test_case_13(self):
        """确定性输入：采样把原始输入规整为确定性快照"""
        raw = {"p1": {"x": 0.123456789, "buttons": ["b", "a"]}, "p2": None}
        first = deterministic_input(raw)
        second = deterministic_input(raw)
        assert first == second
        assert first["p1"] == {"x": 0.1235, "y": 0.0, "buttons": ("a", "b")}
        assert first["p2"] == {"x": 0.0, "y": 0.0, "buttons": ()}

    def test_case_14(self):
        """确定性输入：回放重现输入序列并还原相同最终状态"""
        recorded = FrameSync({"seed": 11})
        run_frames(recorded, 20, {frame: {"p1": {"x": 1.0}, "p2": {"y": -1.0}} for frame in range(20)})
        log = recorded.deterministic_input(replay=True)
        assert len(log) == 20
        assert log[5]["frame"] == 5
        assert log[5]["inputs"]["p1"]["x"] == pytest.approx(1.0)

        replayed = FrameSync({"seed": 11})
        for entry in log:
            replayed.frame_sync(entry["inputs"])
        assert replayed._state == recorded._state

    def test_case_15(self):
        """端到端：帧锁定 + 输入同步 + 确定性物理/渲染整体确定"""
        client_a = FrameSync({"seed": 2024, "frame_rate": 60})
        client_b = FrameSync({"seed": 2024, "frame_rate": 60})
        inputs = {frame: {"p1": {"x": 0.5, "y": 0.25}, "p2": {"x": -0.5}} for frame in range(60)}
        for frame in range(60):
            client_a.frame_sync(inputs[frame])
            client_b.frame_sync(inputs[frame])
        assert client_a._state == client_b._state
        assert client_a._state["frame"] == 60
        render_a = client_a.deterministic_rendering()
        render_b = client_b.deterministic_rendering()
        assert render_a == render_b
        assert client_a.deterministic_random(count=5) == client_b.deterministic_random(count=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
