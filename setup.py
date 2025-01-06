from setuptools import find_packages, setup

setup(
    name="medicalchatbot",
    version="0.0.0",
    author="Larue Linder",
    author_email="laruelinder77@gmail.com",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers==2.2.2",
        "langchain",
        "flask",
        "pypdf",
        "python-dotenv",
        "pinecone[grpc]",
    ]
)