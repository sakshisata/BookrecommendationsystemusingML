from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Books-Recommender-System-Using-Machine-Learning",
    version="0.0.1",
    author="Sakshisata",
    description="A small package for Book Recommender System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sakshisata/Books-Recommender-System-Using-Machine-Learning",
    author_email="business.sakshi05@gmail.com",
    packages=["src"],
    license="MIT",
    python_requires=">=3.7",
    install_requires=['streamlit', 'numpy']
)

