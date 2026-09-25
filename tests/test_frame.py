import pytest

from src import FrameSyncBattle, Fighter


def make_battle(seed=42):
    b = FrameSyncBattle(seed=seed)
    b.add_fighter(Fighter("A", hp=100, max_hp=100, attack=30, defense=10, crit_rate=20, crit_dmg=150))
    b.add_fighter(Fighter("B", hp=100, max_hp=100, attack=25, defense=20, crit_rate=0, crit_dmg=100))
    return b


def test_deterministic_same_seed():
    """相同种子相同输入必须产生相同结果。"""
    b1 = make_battle()
    b2 = make_battle()
    for _ in range(10):
        b1.tick([{"type": "attack", "from": "A", "to": "B"}])
        b2.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b1.fighters["B"].hp == b2.fighters["B"].hp
    assert b1.log == b2.log
    assert b1.serialize() == b2.serialize()


def test_integer_damage_no_float():
    """伤害必须是整数，且等于整数公式推导值。"""
    b = make_battle()
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    for line in b.log:
        dmg = int(line.split("for ")[1])
        assert isinstance(dmg, int)
    # 30 * (100 - 20) // 100 = 24，无暴击（seed=42 首个 roll=73 >= 20）
    assert b.fighters["B"].hp == 100 - 24
    assert isinstance(b.fighters["B"].hp, int)


def test_serialize_deserialize_consistent():
    """序列化后继续战斗结果必须一致。"""
    b1 = make_battle()
    for _ in range(5):
        b1.tick([{"type": "attack", "from": "A", "to": "B"}])
    data = b1.serialize()
    b2 = FrameSyncBattle.deserialize(data)
    assert b2.serialize() == data
    for _ in range(5):
        b1.tick([{"type": "attack", "from": "B", "to": "A"}])
        b2.tick([{"type": "attack", "from": "B", "to": "A"}])
    assert b1.fighters["A"].hp == b2.fighters["A"].hp
    assert b1.serialize() == b2.serialize()


def test_frame_based_not_realtime():
    """时间推进基于帧数，与真实时间无关。"""
    b = make_battle()
    assert b.frame == 0
    b.tick([])
    assert b.frame == 1
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.frame == 2


def test_input_order():
    """同一帧内输入必须按顺序应用。"""
    b = make_battle()
    b.fighters["B"].hp = 10
    b.tick([
        {"type": "attack", "from": "A", "to": "B"},
        {"type": "attack", "from": "B", "to": "A"},
    ])
    assert b.fighters["B"].hp == 0
    # 日志顺序必须与输入顺序一致：A 先打 B，B 后打 A
    assert b.log[0].startswith("frame1 A hits B")
    assert b.log[1].startswith("frame1 B hits A")


def test_rng_consumed_per_attack():
    """每次攻击消耗一个随机数，推进 RNG 状态。"""
    b = make_battle()
    state_before = b.rng_state
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.rng_state != state_before, "每次攻击应推进RNG状态"
    expected = (state_before * 1664525 + 1013904223) % (2 ** 32)
    assert b.rng_state == expected


def test_hp_never_negative():
    """HP 不能为负。"""
    b = make_battle()
    b.fighters["B"].hp = 1
    for _ in range(5):
        b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.fighters["B"].hp >= 0
    assert b.fighters["B"].hp == 0


def test_multiple_fighters_independent():
    """多个战士状态独立。"""
    b = make_battle()
    b.add_fighter(Fighter("C", hp=80, max_hp=80, attack=20, defense=5, crit_rate=50, crit_dmg=200))
    b.tick([{"type": "attack", "from": "C", "to": "A"}])
    assert b.fighters["A"].hp < 100
    assert b.fighters["C"].hp == 80


def test_crit_damage_integer():
    """暴击伤害必须是整数：30 * 150 // 100 = 45。"""
    b = FrameSyncBattle(seed=1)
    b.add_fighter(Fighter("A", hp=100, max_hp=100, attack=30, defense=0, crit_rate=100, crit_dmg=150))
    b.add_fighter(Fighter("B", hp=1000, max_hp=1000, attack=1, defense=0, crit_rate=0, crit_dmg=100))
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    dmg = 1000 - b.fighters["B"].hp
    assert dmg == 45
    assert isinstance(dmg, int)


def test_log_deterministic():
    """相同种子和输入必须产生完全相同的战斗日志。"""
    b1 = make_battle()
    b2 = make_battle()
    for _ in range(3):
        b1.tick([{"type": "attack", "from": "A", "to": "B"}])
        b2.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b1.log == b2.log
    assert len(b1.log) == 3


def test_lcg_sequence_cross_platform():
    """LCG 序列与平台无关：硬编码期望值，任何机器上必须一致。"""
    b = FrameSyncBattle(seed=42)
    expected = [1083814273, 378494188, 2479403867, 955863294, 1613448261, 110225632]
    got = [b._next_rng() for _ in range(6)]
    assert got == expected
    # 相同种子重放得到相同序列
    b2 = FrameSyncBattle(seed=42)
    assert [b2._next_rng() for _ in range(6)] == expected
    # 不同种子序列不同
    b3 = FrameSyncBattle(seed=43)
    assert b3._next_rng() != expected[0]


def test_replay_to_frame_matches_live():
    """回放到指定帧的状态必须与实时战斗到该帧完全一致。"""
    script = [
        [{"type": "attack", "from": "A", "to": "B"}],
        [{"type": "attack", "from": "B", "to": "A"}, {"type": "heal", "to": "B", "amount": 5}],
        [{"type": "modify", "to": "A", "attr": "attack", "percent": 150}],
        [{"type": "attack", "from": "A", "to": "B"}],
        [],
        [{"type": "attack", "from": "A", "to": "B"}, {"type": "attack", "from": "B", "to": "A"}],
    ]
    live = make_battle(seed=7)
    snapshots = []
    for inputs in script:
        live.tick(inputs)
        snapshots.append(live.serialize())
    # 回放到每个中间帧，必须与实时快照一致
    for target in (1, 3, 5, 6):
        replay = make_battle(seed=7)
        for inputs in script[:target]:
            replay.tick(inputs)
        assert replay.serialize() == snapshots[target - 1]
        assert replay.frame == target
        assert replay.log == live.log[: len(replay.log)]


def test_float_inputs_rejected():
    """浮点数禁止参与状态变更：浮点输入必须被拒绝。"""
    b = make_battle()
    with pytest.raises(TypeError):
        b.tick([{"type": "heal", "to": "A", "amount": 10.5}])
    with pytest.raises(TypeError):
        b.tick([{"type": "modify", "to": "A", "attr": "attack", "percent": 150.0}])
    with pytest.raises(TypeError):
        FrameSyncBattle(seed=1.5)
    # 被拒绝的输入不得改变状态（heal 在 tick 内校验失败，A 的 hp 不变）
    assert b.fighters["A"].hp == 100


def test_display_float_isolated():
    """浮点只能用于显示：hp_ratio 返回浮点但不影响任何逻辑状态。"""
    b = make_battle()
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    before = b.serialize()
    ratio = b.hp_ratio("B")
    assert isinstance(ratio, float)
    assert ratio == b.fighters["B"].hp / b.fighters["B"].max_hp
    assert b.serialize() == before, "显示用浮点不得改变战斗状态"


def test_heal_integer_and_clamp():
    """治疗用整数运算，且不超过 max_hp。"""
    b = make_battle()
    b.fighters["A"].hp = 40
    b.tick([{"type": "heal", "to": "A", "amount": 30}])
    assert b.fighters["A"].hp == 70
    assert isinstance(b.fighters["A"].hp, int)
    # 超出 max_hp 部分被截断，日志记录实际治疗量
    b.tick([{"type": "heal", "to": "A", "amount": 999}])
    assert b.fighters["A"].hp == 100
    assert b.log[-1] == "frame2 A healed for 30"


def test_attribute_modifier_integer():
    """属性修正用整数运算，除法整除向下取整。"""
    b = make_battle()
    b.tick([{"type": "modify", "to": "A", "attr": "attack", "percent": 150}])
    assert b.fighters["A"].attack == 45  # 30 * 150 // 100
    b.tick([{"type": "modify", "to": "A", "attr": "attack", "percent": 33}])
    assert b.fighters["A"].attack == 14  # 45 * 33 // 100 = 14（向下取整）
    assert isinstance(b.fighters["A"].attack, int)
    # 修正后的属性立即影响后续伤害计算
    b2 = make_battle()
    b2.tick([{"type": "modify", "to": "A", "attr": "attack", "percent": 200}])
    b2.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b2.fighters["B"].hp == 100 - 48  # 60 * 80 // 100


def test_long_run_accumulation():
    """大量帧累积后两个客户端状态仍完全一致（无累积误差）。"""
    b1 = make_battle(seed=99)
    b2 = make_battle(seed=99)
    for i in range(1000):
        inputs = [
            {"type": "attack", "from": "A", "to": "B"},
            {"type": "attack", "from": "B", "to": "A"},
        ]
        if i % 7 == 0:
            inputs.append({"type": "heal", "to": "A", "amount": 3})
        if i % 11 == 0:
            inputs.append({"type": "modify", "to": "B", "attr": "defense", "percent": 99})
        b1.tick(inputs)
        b2.tick(inputs)
    assert b1.serialize() == b2.serialize()
    assert b1.frame == 1000
    assert b1.log == b2.log
    # 中途存档读档后继续，结果不变
    b3 = make_battle(seed=99)
    for i in range(500):
        b3.tick([{"type": "attack", "from": "A", "to": "B"}, {"type": "attack", "from": "B", "to": "A"}])
    mid = FrameSyncBattle.deserialize(b3.serialize())
    assert mid.serialize() == b3.serialize()


def test_boundary_damage():
    """边界伤害：defense>=100 或 attack=0 时伤害为 0，不为负、不治疗。"""
    b = make_battle()
    b.fighters["B"].defense = 100
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.fighters["B"].hp == 100  # 30 * 0 // 100 = 0
    b.fighters["B"].defense = 150  # 超额防御不得产生负伤害（即回血）
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.fighters["B"].hp == 100
    b.fighters["A"].attack = 0
    b.fighters["B"].defense = 0
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.fighters["B"].hp == 100
    assert b.log[-1].endswith("for 0")


def test_no_float_in_serialized_state():
    """序列化后的战斗状态不得包含任何浮点数。"""
    b = make_battle()
    for _ in range(20):
        b.tick([
            {"type": "attack", "from": "A", "to": "B"},
            {"type": "heal", "to": "B", "amount": 2},
            {"type": "modify", "to": "A", "attr": "attack", "percent": 101},
        ])
    state = b.serialize()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            assert not isinstance(node, float), f"状态中出现浮点: {node!r}"

    walk(state)
    assert isinstance(state["frame"], int)
    assert isinstance(state["rng_state"], int)
