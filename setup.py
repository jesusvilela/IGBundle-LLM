from setuptools import setup, find_packages

setup(
    name="igbundle-llm",
    version="0.1.0",
    description="Information-Geometric Bundle Adapter for LLMs",
    author="Jesús Vilela Jato",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.8",
)
