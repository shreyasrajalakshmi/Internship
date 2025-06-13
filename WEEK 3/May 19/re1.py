import ast
import pkg_resources

# File to analyze
INPUT_FILE = "web3.py"
OUTPUT_FILE = "requirements.txt"

# Step 1: Parse imports
with open(INPUT_FILE, "r") as f:
    tree = ast.parse(f.read())

imports = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for n in node.names:
            imports.add(n.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            imports.add(node.module.split('.')[0])

# Step 2: Get installed version info
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
requirements = []

for imp in sorted(imports):
    pkg_name = imp.lower()
    if pkg_name in installed_packages:
        version = installed_packages[pkg_name]
        requirements.append(f"{pkg_name}=={version}")
    else:
        requirements.append(pkg_name)  # fallback

# Step 3: Write to requirements.txt
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(requirements))

print(f"Generated {OUTPUT_FILE} with {len(requirements)} packages.")
