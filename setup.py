from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

setup(
    name="denoiser",
    version="0.1",
    author="Samuel Blake",
    author_email="samuelfblake@gmail.com",
    description="Audio denoising",
    url="https://github.com/sfblake/recipe-allocator",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['denoiser=denoiser.clean_audio_file:main']
    }
)
