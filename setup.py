from setuptools import setup, find_packages

setup(
    name="igbundle-llm",
    version="0.1.0",
    description="Information-Geometric Bundle Adapter for LLMs",
    author="Jesús Vilela Jato",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.40.0",
        "numpy",
    ],
    python_requires=">=3.8",
)
