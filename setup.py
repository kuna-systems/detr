import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detr-kunasystems",
    version="0.0.3",
    author="detr",
    author_email="",
    description="DETR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kuna-systems/detr",
    project_urls={
        "url": "https://github.com/kuna-systems/detr",
    },
    classifiers=[
    ],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
