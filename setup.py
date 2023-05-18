#!/usr/bin/env python3

import codecs
import os
import platform
import re
from glob import glob
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


def find_version(*file_paths):
    here = os.path.abspath(os.path.dirname(__file__))
    def read(*parts):
        with codecs.open(os.path.join(here, *parts), "r") as fp:
            return fp.read()
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def fetch_requirements():
    requirements_file = "requirements.txt"
    if platform.system() == "Windows":
        DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")
    with open(requirements_file) as f:
        reqs = f.read()
    reqs = reqs.strip().split("\n")
    reqs = remove_specific_requirements(reqs)
    return reqs


def remove_specific_requirements(reqs):
    rtd = "READTHEDOCS" in os.environ
    excluded = {"fasttext": rtd}
    updated_reqs = []
    for req in reqs:
        without_version = req.split("==")[0]
        if not excluded.get(without_version, False):
            updated_reqs.append(req)
    return updated_reqs


def fetch_files_from_folder(folder):
    options = glob(f"{folder}/**", recursive=True)
    data_files = []
    for option in options:
        if os.path.isdir(option):
            files = []
            for f in glob(os.path.join(option, "*")):
                if os.path.isfile(f):
                    files.append(f)
                data_files += files
    return data_files


def fetch_package_data():
    current_dir = os.getcwd()
    captionvqa_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(captionvqa_folder, "captionvqa"))
    data_files = fetch_files_from_folder("projects")
    data_files += fetch_files_from_folder("tools")
    data_files += fetch_files_from_folder("configs")
    data_files += glob(os.path.join("utils", "phoc", "cphoc.*"))
    os.chdir(current_dir)
    return data_files


DISTNAME = "captionvqa"
DEPENDENCY_LINKS = []
REQUIREMENTS = (fetch_requirements(),)

EXCLUDES = ("data", "docs", "tests", "tests.*", "tools", "tools.*")
CMD_CLASS = {"build_ext": build_ext}
EXT_MODULES = [
    Extension(
        "captionvqa.utils.phoc.cphoc", sources=["captionvqa/utils/phoc/src/cphoc.c"], language="c"
    )
]

if "READTHEDOCS" in os.environ:
    EXT_MODULES = []
    CMD_CLASS.pop("build_ext", None)
    DEPENDENCY_LINKS.append(
        "https://download.pytorch.org/whl/cpu/torch-1.5.0%2B"
        + "cpu-cp36-cp36m-linux_x86_64.whl"
    )


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        include_package_data=True,
        package_data={"captionvqa": fetch_package_data()},
        packages=setuptools.find_packages(exclude=EXCLUDES),
        python_requires=">=3.6",
        ext_modules=EXT_MODULES,
        cmdclass=CMD_CLASS,
        version=find_version("captionvqa", "version.py"),
        dependency_links=DEPENDENCY_LINKS,
        classifiers=[
            "Programming Language :: Python :: 3.6"
        ],
        entry_points={
            "console_scripts": [
                "captionvqa_run = captionvqa_cli.run:run",
                "captionvqa_predict = captionvqa_cli.predict:predict",
                "captionvqa_convert_hm = captionvqa_cli.hm_convert:main",
                "captionvqa_interactive = captionvqa_cli.interactive:interactive",
            ]
        },
    )
