# from beartype.claw import beartype_this_package  # <-- boilerplate for victory

# beartype_this_package()
from beartype.claw import beartype_package
from beartype import BeartypeConf

# Exclude the third_party directory by not applying beartype on it
submodules_to_check: list[str] = [
    "monopriors.relative_depth_models",
    # add other submodules except the ones in third_party
]

conf = BeartypeConf()  # Customize the configuration if needed

for submodule in submodules_to_check:
    beartype_package(submodule, conf=conf)
