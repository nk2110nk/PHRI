import subprocess

# findでパス取得
result = subprocess.run(
    ["find", "results_case4", "-mindepth", "2", "-maxdepth", "2", "-type", "d"],
    stdout=subprocess.PIPE,
    text=True
)

paths = result.stdout.strip().split("\n")

commands = []

for p in paths:
    if not p.strip():
        continue

    parts = p.split("/")
    domain_algo = parts[1]  # 例: Car_Boulware-Atlas3

    # domain抽出（EnergySmall_A対応）
    domain_parts = domain_algo.split("_")
    domain = "_".join(domain_parts[:-1])
    algo_part = domain_parts[-1]

    # 複数エージェント対応
    algos = algo_part.split("-")
    algo_str = " ".join(algos)

    cmd = f"python3 ./test_negotiator.py -a {algo_str} -i {domain} -m ./{p}/"
    commands.append(cmd)

# .shとして出力
with open("run_test4.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for c in commands:
        f.write(c + "\n")

print(f"{len(commands)}件のコマンドを run_test1.sh に出力しました")