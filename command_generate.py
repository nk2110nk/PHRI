import argparse
import os
import re
import shlex
from pathlib import Path


KNOWN_AGENTS = {
    "Boulware",
    "Linear",
    "Conceder",
    "TitForTat1",
    "TitForTat2",
    "AgentK",
    "HardHeaded",
    "Atlas3",
    "AgentGG",
}


def load_domain_names(domain_root):
    return {
        path.name
        for path in domain_root.iterdir()
        if path.is_dir() and (path / "domain.xml").exists()
    }


def split_model_dir_name(name, domain_names):
    for index, char in enumerate(name):
        if char != "_":
            continue

        domain_part = name[:index]
        agent_part = name[index + 1 :]
        domains = domain_part.split("-")
        agents = agent_part.split("-")

        if (
            domains
            and agents
            and all(domain in domain_names for domain in domains)
            and all(agent in KNOWN_AGENTS for agent in agents)
        ):
            return domains, agents

    raise ValueError(f"results directory name could not be parsed: {name}")


def find_model_dirs(results_root):
    checkpoints = results_root.glob("*/**/MiPN_Negotiator/checkpoint.pt")
    return sorted(checkpoint.parent for checkpoint in checkpoints)


def build_command(model_dir, domains, agents):
    model_path = model_dir.as_posix()
    if not model_dir.is_absolute():
        model_path = f"./{model_path}"

    args = [
        "python3",
        "./test_negotiator.py",
        "-a",
        *agents,
        "-i",
        *domains,
        "-m",
        f"{model_path}/",
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def default_output_path(results_root):
    match = re.fullmatch(r"results_case(\d+)", results_root.name)
    if match:
        return Path(f"run_test{match.group(1)}.sh")
    return Path("run_test.sh")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir_arg", nargs="?")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--domain-dir", default="domain")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    results_root = Path(args.results_dir_arg or args.results_dir)
    domain_root = Path(args.domain_dir)
    if not results_root.is_dir():
        raise SystemExit(f"results directory does not exist: {results_root}")

    domain_names = load_domain_names(domain_root)

    commands = []
    for model_dir in find_model_dirs(results_root):
        model_group = model_dir.parents[1].name
        domains, agents = split_model_dir_name(model_group, domain_names)
        commands.append(build_command(model_dir, domains, agents))

    output_path = Path(args.output) if args.output else default_output_path(results_root)
    with output_path.open("w") as f:
        f.write("#!/bin/bash\n\n")
        for command in commands:
            f.write(command + "\n")

    os.chmod(output_path, 0o755)
    print(f"{len(commands)} commands written to {output_path}")


if __name__ == "__main__":
    main()
