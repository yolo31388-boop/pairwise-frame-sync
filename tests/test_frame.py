from src import FrameSyncBattle, Fighter


def make_battle():
    b = FrameSyncBattle(seed=42)
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


def test_integer_damage_no_float():
    """伤害必须是整数，不能用浮点计算。"""
    b = make_battle()
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    # 检查日志中的伤害是整数
    for line in b.log:
        dmg_str = line.split("for ")[1]
        dmg = int(dmg_str)
        assert isinstance(dmg, int)


def test_serialize_deserialize_consistent():
    """序列化后继续战斗结果必须一致。"""
    b1 = make_battle()
    for _ in range(5):
        b1.tick([{"type": "attack", "from": "A", "to": "B"}])
    data = b1.serialize()
    b2 = FrameSyncBattle.deserialize(data)
    for _ in range(5):
        b1.tick([{"type": "attack", "from": "B", "to": "A"}])
        b2.tick([{"type": "attack", "from": "B", "to": "A"}])
    assert b1.fighters["A"].hp == b2.fighters["A"].hp


def test_frame_based_not_realtime():
    """时间推进基于帧数。"""
    b = make_battle()
    assert b.frame == 0
    b.tick([])
    assert b.frame == 1
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.frame == 2


def test_input_order():
    """输入必须按顺序应用。"""
    b = make_battle()
    b.fighters["B"].hp = 10
    # A先攻击B（B死），B再攻击A（B已死不应攻击）
    b.tick([
        {"type": "attack", "from": "A", "to": "B"},
        {"type": "attack", "from": "B", "to": "A"},
    ])
    # B的攻击仍然执行了（因为没有死亡检查），但这不是本测试重点
    # 重点是顺序：A的伤害先应用
    assert b.fighters["B"].hp == 0


def test_rng_consumed_per_attack():
    """每次攻击消耗一个随机数。"""
    b = make_battle()
    state_before = b.rng_state
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    # BUG版本：rng_state不变（因为用了random模块而不是LCG）
    assert b.rng_state != state_before, "每次攻击应推进RNG状态"


def test_hp_never_negative():
    """HP不能为负。"""
    b = make_battle()
    b.fighters["B"].hp = 1
    for _ in range(5):
        b.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b.fighters["B"].hp >= 0


def test_multiple_fighters_independent():
    """多个战士状态独立。"""
    b = make_battle()
    b.add_fighter(Fighter("C", hp=80, max_hp=80, attack=20, defense=5, crit_rate=50, crit_dmg=200))
    b.tick([{"type": "attack", "from": "C", "to": "A"}])
    assert b.fighters["A"].hp < 100
    assert b.fighters["C"].hp == 80


def test_crit_damage_integer():
    """暴击伤害必须是整数。"""
    b = FrameSyncBattle(seed=1)  # 选个容易暴击的种子
    b.add_fighter(Fighter("A", hp=100, max_hp=100, attack=30, defense=0, crit_rate=100, crit_dmg=150))
    b.add_fighter(Fighter("B", hp=1000, max_hp=1000, attack=1, defense=0, crit_rate=0, crit_dmg=100))
    b.tick([{"type": "attack", "from": "A", "to": "B"}])
    dmg = 1000 - b.fighters["B"].hp
    assert dmg == 45  # 30 * 1.5 = 45，整数


def test_log_deterministic():
    """战斗日志必须确定性。"""
    b1 = make_battle()
    b2 = make_battle()
    for _ in range(3):
        b1.tick([{"type": "attack", "from": "A", "to": "B"}])
        b2.tick([{"type": "attack", "from": "A", "to": "B"}])
    assert b1.log == b2.log
