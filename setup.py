import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="biothings-idlookup",
    version="0.0.2",
    packages=setuptools.find_packages(),
    install_requires=[
        'biothings_client>=0.2.1',
    ]
)
