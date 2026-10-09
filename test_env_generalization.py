import os, importlib, sys
sys.path.insert(0, ".")

def check(net_path, expected_agents):
    os.environ["MARL_NET_FILE"] = net_path
    import env.traci_env as te
    importlib.reload(te)
    ok_lanes = all(len(v) == 4 for v in te.LANE_IDS.values())
    print(f"{net_path}")
    print(f"  agents: {te.NUM_AGENTS} (expected {expected_agents})")
    print(f"  sample: {te.TL_IDS[:4]}")
    print(f"  every light has 4 lanes: {ok_lanes}")
    print(f"  A0 lanes: {te.LANE_IDS.get('A0')}")
    assert te.NUM_AGENTS == expected_agents
    assert ok_lanes
    print("  PASS\n")

check("configs/mygrid2x2.net.xml", 4)
check("configs/mygrid4x4.net.xml", 16)
print("Both grids derive correctly.")