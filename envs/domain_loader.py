from pathlib import Path

from negmas import Issue, UtilityFunction


DOMAIN_ROOT = Path(__file__).resolve().parents[1] / "domain"

DOMAIN_FILE_NAMES = {
    "Coffee": "Coffee.xml",
    "Camera": "Camera-A-domain.xml",
    "Lunch": "Lunch.xml",
    "SmartPhone": "SmartPhone.xml",
    "Kitchen": "Kitchen-domain.xml",
}

UTILITY_FILE_NAMES = {
    "Coffee": ("Coffee_util1.xml", "Coffee_util2.xml", "Coffee_util3.xml"),
    "Camera": ("Camera-A-prof1.xml", "Camera-A-prof2.xml", "Camera-A-prof3.xml"),
    "Lunch": ("Lunch_util1.xml", "Lunch_util2.xml", "Lunch_util3.xml"),
    "SmartPhone": ("SmartPhone_util1.xml", "SmartPhone_util2.xml", "SmartPhone_util3.xml"),
    "Kitchen": ("Kitchen-husband.xml", "Kitchen-wife.xml", "Kitchen-children.xml"),
}


def domain_file_path(domain_name):
    domain_dir = DOMAIN_ROOT / domain_name
    default_path = domain_dir / "domain.xml"
    if default_path.exists():
        return default_path

    file_name = DOMAIN_FILE_NAMES.get(domain_name)
    if file_name is None:
        raise FileNotFoundError(f"Cannot find domain file for {domain_name}: {default_path}")

    path = domain_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find domain file for {domain_name}: {path}")
    return path


def utility_file_path(domain_name, scenario_number):
    domain_dir = DOMAIN_ROOT / domain_name
    default_path = domain_dir / f"utility{scenario_number + 1}.xml"
    if default_path.exists():
        return default_path

    file_names = UTILITY_FILE_NAMES.get(domain_name)
    if file_names is None or scenario_number >= len(file_names):
        raise FileNotFoundError(f"Cannot find utility file for {domain_name}: {default_path}")

    path = domain_dir / file_names[scenario_number]
    if not path.exists():
        raise FileNotFoundError(f"Cannot find utility file for {domain_name}: {path}")
    return path


def load_domain(domain_name):
    return Issue.from_genius(str(domain_file_path(domain_name)))


def load_utility(domain_name, scenario_number):
    return UtilityFunction.from_genius(str(utility_file_path(domain_name, scenario_number)))


def load_genius_domain(domain_name, scenario_numbers=(0, 1, 2)):
    domain, _ = load_domain(domain_name)
    utilities = [load_utility(domain_name, scenario_number)[0] for scenario_number in scenario_numbers]
    return domain, utilities
