"""帧同步不变量测试：整数运算、确定性RNG、序列化、回放、浮点隔离。"""

import json

from src import FrameSyncBattle, Fighter


def make_battle(seed=42):
    b = FrameSyncBattle(seed=seed)
    b.add_fighter(Fighter("A", hp=100, max_hp=100, attack=30, defense=10, crit_rate=20, crit_dmg=150))
    b.add_fighter(Fighter("B", hp=100, max_hp=100, attack=25, defense=20, crit_rate=0, crit_dmg=100))
    return b


def run_script(b, frames):
    for inputs in frames:
        b.tick(inputs)
    return b


SCRIPT = [
    [{"type": "attack", "from": "A", "to": "B"}],
    [{"type": "attack", "from": "B", "to": "A"}, {"type": "attack", "from": "A", "to": "B"}],
    [{"type": "heal", "to": "B", "amount": 15}],
    [{"type": "modify", "to": "A", "attr": "attack", "pct": 10}],
    [{"type": "attack", "from": "A", "to": "B"}, {"type": "heal", "to": "A", "amount": 5}],
]


# 1. 整数伤害：伤害公式用整除，结果精确可预期
def test_integer_damage_exact():
    b = FrameSyncBattle(seed=7)
    b.add_fighter(Fighter("X", hp=100, max_hp=100, attack=33, defense=0, crit_rate=0, crit_dmg=100))
    b.add_fighter(Fighter("Y", hp=100, max_hp=100, attack=1, defense=15, crit_rate=0, crit_dmg=100))
    # crit_rate=0 时不暴击？roll<0 永不成立，伤害 = 33*(100-15)//100 = 28
    b.tick([{"type": "attack", "from": "X", "to": "Y"}])
    assert b.fighters["Y"].hp == 100 - 28


# 2. 确定性随机：LCG序列与手算值一致（跨平台一致）
def test_lcg_sequence_cross_platform():
    b = FrameSyncBattle(seed=42)
    # glibc LCG: state = (state*1103515245 + 12345) % 2**31
    assert b._next_rng() == 1250496027
    state = 1250496027
    for _ in range(10):
        state = (state * 1103515245 + 12345) % (2 ** 31)
        assert b._next_rng() == state


# 3. 浮点隔离：显示用浮点不影响状态
def test_float_display_isolated():
    b = make_battle()
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    snapshot = b.serialize()
    pct = b.display_hp_percent("B")
    assert isinstance(pct, float)
    assert b.serialize() == snapshot


# 4. 状态中无浮点：序列化结果JSON往返后无float
def test_no_float_in_state():
    b = make_battle()
    run_script(b, SCRIPT)
    data = json.loads(json.dumps(b.serialize()))

    def walk(o):
        if isinstance(o, float):
            return True
        if isinstance(o, dict):
            return any(walk(v) for v in o.values())
        if isinstance(o, list):
            return any(walk(v) for v in o)
        return False

    assert not walk(data)


# 5. 序列化一致性：序列化/反序列化后继续战斗结果一致
def test_serialize_continue_consistent():
    b1 = make_battle()
    run_script(b1, SCRIPT[:3])
    b2 = FrameSyncBattle.deserialize(b1.serialize())
    run_script(b1, SCRIPT[3:])
    run_script(b2, SCRIPT[3:])
    assert b1.serialize()["fighters"] == b2.serialize()["fighters"]
    assert b1.log == b2.log
    assert b1.rng_state == b2.rng_state


# 6. 帧同步：两个客户端相同输入逐帧同步
def test_frame_sync_two_clients():
    c1, c2 = make_battle(seed=99), make_battle(seed=99)
    for inputs in SCRIPT:
        c1.tick(inputs)
        c2.tick(inputs)
        assert c1.serialize() == c2.serialize()


# 7. 回放一致：回放到指定帧与实时战斗到该帧完全一致
def test_replay_matches_live():
    live = make_battle(seed=123)
    run_script(live, SCRIPT)
    replayed = live.replay_to(3)
    live3 = make_battle(seed=123)
    run_script(live3, SCRIPT[:3])
    assert replayed.serialize()["fighters"] == live3.serialize()["fighters"]
    assert replayed.log == live3.log
    assert replayed.frame == 3


# 8. 输入顺序：同一帧内输入严格按顺序应用
def test_input_order_matters():
    b = make_battle()
    b.fighters["B"].hp = 5
    b.tick([
        {"type": "attack", "from": "A", "to": "B"},
        {"type": "heal", "to": "B", "amount": 50},
    ])
    # 先攻击打死B(hp=0)，再治疗恢复到50；顺序反过来结果不同
    assert b.fighters["B"].hp == 50
    assert "hits" in b.log[0] and "healed" in b.log[1]


# 9. 时间基于帧：frame只随tick推进
def test_time_is_frame_based():
    import time
    b = make_battle()
    b.tick([])
    time.sleep(0.01)
    assert b.frame == 1
    b.tick([])
    b.tick([])
    assert b.frame == 3


# 10. 相同种子相同结果：完整战斗日志与结果一致
def test_same_seed_same_result():
    b1, b2 = make_battle(seed=2024), make_battle(seed=2024)
    run_script(b1, SCRIPT)
    run_script(b2, SCRIPT)
    assert b1.log == b2.log
    assert b1.serialize() == b2.serialize()


# 11. 不同种子结果不同（RNG确实参与）
def test_different_seed_diverges():
    b1 = make_battle(seed=1)
    b2 = make_battle(seed=2)
    frames = [[{"type": "attack", "from": "A", "to": "B"}] for _ in range(20)]
    run_script(b1, frames)
    run_script(b2, frames)
    assert b1.log != b2.log or b1.fighters["B"].hp != b2.fighters["B"].hp


# 12. 大量帧累积：1000帧后两客户端仍完全一致（无累积误差）
def test_long_run_no_drift():
    c1, c2 = make_battle(seed=555), make_battle(seed=555)
    for i in range(1000):
        inputs = [{"type": "attack", "from": "A" if i % 2 == 0 else "B",
                   "to": "B" if i % 2 == 0 else "A"}]
        if i % 10 == 0:
            inputs.append({"type": "heal", "to": "A", "amount": 3})
        c1.tick(inputs)
        c2.tick(inputs)
    assert c1.serialize() == c2.serialize()
    assert c1.frame == 1000


# 13. 边界伤害：防御>=100时伤害为0，HP不为负
def test_boundary_damage():
    b = FrameSyncBattle(seed=3)
    b.add_fighter(Fighter("X", hp=10, max_hp=10, attack=50, defense=0, crit_rate=0, crit_dmg=100))
    b.add_fighter(Fighter("Y", hp=5, max_hp=5, attack=1, defense=100, crit_rate=0, crit_dmg=100))
    b.tick([{"type": "attack", "from": "X", "to": "Y"}])
    assert b.fighters["Y"].hp == 5  # 50*0//100 = 0伤害
    b.fighters["Y"].defense = 0
    for _ in range(5):
        b.tick([{"type": "attack", "from": "X", "to": "Y"}])
    assert b.fighters["Y"].hp == 0


# 14. 治疗计算：整数治疗且不超过max_hp
def test_heal_integer_and_capped():
    b = make_battle()
    b.fighters["A"].hp = 90
    b.tick([{"type": "heal", "to": "A", "amount": 25}])
    assert b.fighters["A"].hp == 100  # 封顶max_hp
    b.tick([{"type": "heal", "to": "A", "amount": 7}])
    assert b.fighters["A"].hp == 100
    assert "healed for 10" in b.log[0]
    assert "healed for 0" in b.log[1]


# 15. 属性修正：百分比与增量都用整数运算（向下取整）
def test_attribute_modify_integer():
    b = make_battle()
    b.fighters["A"].attack = 33
    b.tick([{"type": "modify", "to": "A", "attr": "attack", "pct": 10}])
    assert b.fighters["A"].attack == 33 * 110 // 100  # 36，向下取整
    b.tick([{"type": "modify", "to": "A", "attr": "defense", "delta": 5}])
    assert b.fighters["A"].defense == 15
    b.tick([{"type": "modify", "to": "A", "attr": "attack", "pct": -50}])
    assert b.fighters["A"].attack == 36 * 50 // 100  # 18


# 16. 战斗日志：确定性且记录帧号与数值
def test_battle_log_deterministic_and_complete():
    b1, b2 = make_battle(seed=8), make_battle(seed=8)
    run_script(b1, SCRIPT)
    run_script(b2, SCRIPT)
    assert b1.log == b2.log
    assert len(b1.log) > 0
    for line in b1.log:
        assert line.startswith("frame")
        # 日志中所有数值都是整数字符串
        for token in line.split():
            if token.lstrip("-").isdigit():
                int(token)  # 必须能解析为整数


# 17. 暴击判定确定性：同一帧同一攻击者暴击结果固定
def test_crit_roll_deterministic():
    results = []
    for _ in range(2):
        b = FrameSyncBattle(seed=42)
        b.add_fighter(Fighter("A", hp=100, max_hp=100, attack=30, defense=0, crit_rate=50, crit_dmg=200))
        b.add_fighter(Fighter("B", hp=10000, max_hp=10000, attack=1, defense=0, crit_rate=0, crit_dmg=100))
        for _ in range(10):
            b.tick([{"type": "attack", "from": "A", "to": "B"}])
        results.append([line.rsplit(" ", 1)[1] for line in b.log])
    assert results[0] == results[1]
    # 暴击(60)与非暴击(30)都应出现，证明RNG既确定又真实参与
    assert "60" in results[0] and "30" in results[0]


# 18. RNG状态随攻击推进且可序列化恢复
def test_rng_state_advances_and_restores():
    b = make_battle()
    s0 = b.rng_state
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.rng_state != s0
    b2 = FrameSyncBattle.deserialize(b.serialize())
    assert b2.rng_state == b.rng_state
    # 恢复后下一次随机数一致
    assert b2._next_rng() == b._next_rng()
