import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LindbladQD",
    version="0.0.1",
    author="R K Rupesh and Zoltán Guba",
    author_email="gubazoltan99@gmail.com",
    description="This package is used to simulate the dynamics of a single electron in a double quantum dot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thundergoth/LindbladDynamics",
    python_requires=">=3.8",
)
