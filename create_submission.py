import re
import os
import shutil
import argparse
from glob import glob

VERSIONS_DIR = "versions"


def write_file(file_name, out_file):
    out_file.write("#" * 40 + "\n")
    out_file.write(
        "### " + os.path.splitext(os.path.basename(file_name))[0].upper() + "\n"
    )
    out_file.write("#" * 40 + "\n")

    with open(file_name, "r") as file:
        for line in file.readlines():
            if re.match(
                "from \..* import .*|from src import .*|from src.* import .*", line
            ):
                line = f"# {line}"

            if line.startswith("IS_KAGGLE = False"):
                line = "IS_KAGGLE = True\n"

            if line.startswith("LEVEL = logging."):
                line = "LEVEL = logging.INFO\n"

            if line.startswith("PLOT_BOARD = "):
                line = "PLOT_BOARD = False\n"

            out_file.write(line)

    out_file.write("\n\n\n")


def main(out_file_name):
    if os.path.exists("_versions"):
        shutil.rmtree("_versions")

    if not os.path.exists(VERSIONS_DIR):
        os.mkdir(VERSIONS_DIR)

    base_name = os.path.splitext(out_file_name)[0]
    source_dir = os.path.join(VERSIONS_DIR, base_name)
    submission_file = os.path.join(VERSIONS_DIR, out_file_name)
    if os.path.exists(source_dir) or os.path.exists(submission_file):
        print(f"Version '{out_file_name}' already exists.")
        return

    shutil.copytree("src/", os.path.join(source_dir, "src/"))
    shutil.copyfile("agent.py", os.path.join(source_dir, "agent.py"))

    with open(submission_file, "w") as out_file:
        files = ["logger.py", "portion.py", "geometry.py", "models.py", "board.py"]
        for file in files:
            write_file("src/" + file, out_file)

        for file_name in glob("src/*.py"):
            if os.path.basename(file_name) in files:
                continue

            write_file(file_name, out_file)

        write_file("agent.py", out_file)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="submission name")
    flags = parser.parse_args()
    main(flags.name)
