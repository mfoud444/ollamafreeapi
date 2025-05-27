import re
import sys
from pathlib import Path

def update_version(new_version):
    # Update version.py
    version_file = Path("ollamafreeapi/version.py")
    version_file.write_text(f'VERSION = "{new_version}"\n')
    
    # Update pyproject.toml
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    content = re.sub(r'version = ".*?"', f'version = "{new_version}"', content)
    pyproject_file.write_text(content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_version.py <new_version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    update_version(new_version)
    print(f"Updated version to {new_version} in all files") 