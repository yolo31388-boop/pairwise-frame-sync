"""
游戏内帧同步与确定性渲染系统 - 验收测试
运行方式：python -m pytest tests/test_frame_sync.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.frame_sync import FrameSync


class TestFrameSync:
    def setup_method(self):
        self.system = FrameSync()

    # 帧同步：帧锁定——真实时间按固定步长消费，不足一步不推进
    def test_case_01(self):
        fs = FrameSync({"frame_rate": 10})
        for f in range(6):
            fs.submit_input("p1", {"move": f}, frame=f)
        assert fs.frame_dt == pytest.approx(0.1)
        assert fs.update(0.5) == 5
        assert fs.frame == 5
        assert fs.update(0.05) == 0
        assert fs.update(0.05) == 1
        assert fs.frame == 6

    # 帧同步：输入同步——多玩家输入按 id 确定性合并，无输入则停帧等待
    def test_case_02(self):
        fs = self.system
        fs.submit_input("p2", {"move": 1})
        fs.submit_input("p1", {"jump": True})
        merged = fs.step()
        assert list(merged.keys()) == ["p1", "p2"]
        assert merged["p1"] == {"jump": True}
        assert merged["p2"] == {"move": 1}
        assert fs.inputs_ready() is False
        assert fs.update(1.0) == 0
        assert fs.frame == 1

    # 确定性随机数：种子——同种子同结果，不同种子不同结果，可重设种子
    def test_case_03(self):
        a, b, c = FrameSync({"seed": 42}), FrameSync({"seed": 42}), FrameSync({"seed": 7})
        assert a.deterministic_random() == b.deterministic_random()
        assert b.deterministic_random() != c.deterministic_random()
        reseeded = a.deterministic_random(seed=99)
        assert reseeded == FrameSync({"seed": 99}).deterministic_random()

    # 确定性随机数：序列——同种子序列完全一致，值域 [0,1)
    def test_case_04(self):
        s1 = self.system.random_sequence(8, seed=2024)
        s2 = FrameSync({"seed": 2024}).random_sequence(8)
        assert s1 == s2
        assert len(s1) == 8
        assert all(0.0 <= v < 1.0 for v in s1)
        assert len(set(s1)) == 8
        assert FrameSync({"seed": 2025}).random_sequence(8) != s1

    # 确定性渲染：确定性排序——按 (layer, id) 排序，与输入顺序无关
    def test_case_05(self):
        objs = [
            {"id": "c", "layer": 2},
            {"id": "a", "layer": 0},
            {"id": "b", "layer": 0},
        ]
        r1 = self.system.deterministic_rendering(objs)
        r2 = FrameSync().deterministic_rendering(list(reversed(objs)))
        assert [o["id"] for o in r1["objects"]] == ["a", "b", "c"]
        assert [o["id"] for o in r2["objects"]] == ["a", "b", "c"]

    # 确定性渲染：确定性随机数——同种子渲染结果逐字节一致
    def test_case_06(self):
        r1 = FrameSync({"seed": 5}).deterministic_rendering([{"id": "x"}])
        r2 = FrameSync({"seed": 5}).deterministic_rendering([{"id": "x"}])
        r3 = FrameSync({"seed": 6}).deterministic_rendering([{"id": "x"}])
        assert r1 == r2
        assert r1["jitter"] != r3["jitter"]

    # 确定性物理：确定性积分——固定步长积分结果精确且可复现
    def test_case_07(self):
        def run():
            fs = FrameSync({"frame_rate": 10})
            fs.add_body(x=0.0, y=0.0, vx=1.0, vy=-2.0, radius=0.0, body_id="b")
            fs.physics_step()
            return fs.snapshot()
        s1, s2 = run(), run()
        assert s1 == s2
        assert s1["bodies"]["b"]["x"] == pytest.approx(0.1)
        assert s1["bodies"]["b"]["y"] == pytest.approx(-0.2)

    # 确定性物理：确定性碰撞——等质量弹性碰撞交换速度，与插入顺序无关
    def test_case_08(self):
        def run(order):
            fs = FrameSync()
            bodies = [("a", -0.5, 2.0), ("b", 0.5, -2.0)]
            for bid, x, vx in (bodies if order else bodies[::-1]):
                fs.add_body(x=x, y=0.0, vx=vx, vy=0.0, radius=1.0, body_id=bid)
            fs.physics_step(0.1)
            return fs.snapshot()
        s1, s2 = run(True), run(False)
        assert s1 == s2
        assert s1["bodies"]["a"]["vx"] == pytest.approx(-2.0)
        assert s1["bodies"]["b"]["vx"] == pytest.approx(2.0)
        assert s1["bodies"]["a"]["x"] == pytest.approx(-1.0)
        assert s1["bodies"]["b"]["x"] == pytest.approx(1.0)

    # 确定性物理：多步模拟——相同初始条件 50 步后状态完全一致
    def test_case_09(self):
        def run():
            fs = FrameSync({"seed": 9})
            fs.add_body(x=0, y=0, vx=1.0, vy=0.5, body_id=1)
            fs.add_body(x=3, y=0, vx=-1.0, vy=-0.5, body_id=2)
            for _ in range(50):
                fs.physics_step(1 / 30)
            return fs.snapshot()
        assert run() == run()

    # 确定性输入：输入采样——键顺序规范化，采样结果与提交顺序无关
    def test_case_10(self):
        fs = self.system
        sampled = fs.deterministic_input("p1", {"b": 2, "a": 1})
        assert list(sampled.keys()) == ["a", "b"]
        fs.step()
        fs2 = FrameSync()
        fs2.submit_input("p1", {"a": 1, "b": 2})
        fs2.step()
        assert fs.snapshot() == fs2.snapshot()

    # 确定性输入：输入回放——重放输入序列精确复现世界状态
    def test_case_11(self):
        fs = FrameSync({"seed": 3})
        fs.add_body(x=0, y=0, vx=1.0, vy=1.0, body_id="b")
        for f in range(5):
            fs.submit_input("p1", {"move": f}, frame=f)
            fs.step()
        snap = fs.snapshot()
        assert len(fs.input_history()) == 5
        fs2 = FrameSync({"seed": 3})
        fs2.add_body(x=0, y=0, vx=1.0, vy=1.0, body_id="b")
        fs2.replay_inputs(fs.input_history())
        assert fs2.snapshot() == snap
        fs.replay_inputs()
        assert fs.snapshot() == snap

    # 帧同步：跨客户端一致性——相同输入流下两客户端逐帧状态一致
    def test_case_12(self):
        def client():
            fs = FrameSync({"seed": 11, "frame_rate": 30})
            fs.add_body(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=1.0, body_id="a")
            fs.add_body(x=2.5, y=0.0, vx=-1.0, vy=0.0, radius=1.0, body_id="b")
            return fs
        c1, c2 = client(), client()
        for f in range(10):
            for c in (c1, c2):
                c.submit_input("p1", {"move": f % 3}, frame=f)
                c.submit_input("p2", {"move": -(f % 2)}, frame=f)
        renders1, renders2 = [], []
        for _ in range(10):
            assert c1.update(1 / 30) == 1
            assert c2.update(1 / 30) == 1
            renders1.append(c1.deterministic_rendering(
                [{"id": "a", "layer": 0}, {"id": "b", "layer": 0}]))
            renders2.append(c2.deterministic_rendering(
                [{"id": "b", "layer": 0}, {"id": "a", "layer": 0}]))
        assert c1.frame == 10
        assert c1.snapshot() == c2.snapshot()
        assert renders1 == renders2

    # 帧同步：帧率锁定——frame_dt 由帧率决定，update 按帧率切分时间
    def test_case_13(self):
        fs = FrameSync({"frame_rate": 60})
        assert fs.frame_dt == pytest.approx(1 / 60)
        for f in range(3):
            fs.submit_input("p1", {"x": f}, frame=f)
        assert fs.update(1 / 30) == 2
        assert fs.update(1 / 60) == 1
        assert fs.frame == 3

    # 确定性物理：浮点确定性——长序列积分结果逐位相等（非近似）
    def test_case_14(self):
        def simulate():
            fs = FrameSync()
            fs.add_body(x=0.1, y=0.2, vx=0.3, vy=0.7, radius=0.0, body_id="f")
            for _ in range(100):
                fs.physics_step(0.01)
            return fs.snapshot()["bodies"]["f"]
        r1, r2 = simulate(), simulate()
        assert r1["x"] == r2["x"]
        assert r1["y"] == r2["y"]

    # 确定性随机数：跨实例一致性——同种子多实例序列相同，reset 后可重现
    def test_case_15(self):
        seqs = [FrameSync({"seed": 777}).random_sequence(16) for _ in range(3)]
        assert seqs[0] == seqs[1] == seqs[2]
        assert FrameSync({"seed": 1}).random_sequence(16) != \
               FrameSync({"seed": 2}).random_sequence(16)
        fs = FrameSync({"seed": 555})
        first = fs.random_sequence(4)
        fs.reset()
        assert fs.random_sequence(4) == first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
