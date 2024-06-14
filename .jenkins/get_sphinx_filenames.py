from typing import List

from get_files_to_run import get_all_files
from validate_tutorials_built import NOT_RUN


def get_files_for_sphinx() -> List[str]:
    all_py_files = get_all_files()
    return [x for x in all_py_files if all(y not in x for y in NOT_RUN)]


SPHINX_SHOULD_RUN = "|".join(get_files_for_sphinx())

if __name__ == "__main__":
    print(SPHINX_SHOULD_RUN)
