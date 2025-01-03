from cx_Freeze import setup, Executable
import sys
import os
sys.setrecursionlimit(5000)

# Add hozzá az összes szükséges extra fájlt és mappát
include_files = [
    ("images", "images"),
    ("weights", "weights"),
    ("scores", "scores"),
    ("snake", "snake")
]

# Függőségek meghatározása (ha szükséges)
packages = [
    "gym", "pygame", "numpy", "tensorflow", "keras", "matplotlib",
    "customtkinter", "gym.envs.atari", "ale_py", "AutoROM", "snake"
]

# Futtatható fájl definíciója
executables = [
    Executable(
        script="main.py",  # A belépési pont, a fő szkript
        target_name="GamingAI.exe",  # Az exe neve
        base=None  # Konzol nélküli GUI alkalmazás
    )
]

# Setup konfiguráció
setup(
    name="GamingAI",
    version="1.0",
    description="AI Gaming Application",
    options={
        "build_exe": {
            "packages": packages,
            "include_files": include_files,
            "excludes": ["tkinter.test"]  # Nem szükséges modulok kizárása
        }
    },
    executables=executables
)