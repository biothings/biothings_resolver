import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="biothings-resolver",
    version="0.0.3",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'biothings_client>=0.2.1',
    ]
)
