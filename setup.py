import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aim_trajectory_picking",
    version="0.0.1",
    author="Equinor",
    author_email=".@equinor.com",
    description="Trajectory picking algorithm for the AI for Maturation project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/aim-trajectory-picking",
    project_urls={
        "Bug Tracker": "https://github.com/equinor/aim-trajectory-picking/issues",
    },
    entry_points = {
        'console_scripts': ['funniest-joke=aim_trajectory_picking.command_line:main', 
            'run=aim_trajectory_picking.pick_trajectories:main', 
            'ortools=aim_trajectory_picking.ortools_solver:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where='src'),
    package_dir={"": "src"},
    package_data={'': ['benchmark.txt']},
    include_package_data=True,
    install_requires=["networkx", "matplotlib", "numpy", "pandas", "python-igraph","argparse","ortools"],
    python_requires=">=3.6",
)