import os


for name in os.listdir(os.path.dirname(__file__)):
    if os.path.isdir(os.path.join(os.path.dirname(__file__), name)) and name not in [
        "__pycache__",
        ".git",
    ]:
        __import__(".".join(os.path.dirname(__file__).split(os.sep)[-1:] + [name]))
