"""
Settings for MiLP Bayesian Optimization search.
Defines task configurations, objectives, and reference points.
"""

# Task settings for multi-objective optimization
TASK_SETTINGS = {
    "amazonqa": {
        "objectives": ["param", "acc"],
        "ref_point": [100.0, 0.0],
        "patience": 5,
        "num_epochs": 10,
        "save_type": "steps",
        "num_steps_per_save": 500,
    },
    "reddit": {
        "objectives": ["param", "acc"],
        "ref_point": [100.0, 0.0],
        "patience": 5,
        "num_epochs": 10,
        "save_type": "steps",
        "num_steps_per_save": 500,
    },
    "meddia": {
        "objectives": ["param", "acc"],
        "ref_point": [100.0, 0.0],
        "patience": 5,
        "num_epochs": 10,
        "save_type": "steps",
        "num_steps_per_save": 500,
    },
}

# Single-objective task settings
TASK_SETTINGS_SO = {
    "amazonqa": {
        "objectives": ["acc"],
        "ref_point": [0.0],
    },
    "reddit": {
        "objectives": ["acc"],
        "ref_point": [0.0],
    },
    "meddia": {
        "objectives": ["acc"],
        "ref_point": [0.0],
    },
}
