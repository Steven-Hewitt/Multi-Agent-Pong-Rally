# Multi-Agent-Pong-Rally (MAPR)
### A Gym environment.

MAPR is a Gym environment designed to emphasize collaboration between two actors.

Each actor has one action dimension, which represents an acceleration value for their respective paddle on the board.

## Installation Instructions

Each of the files in this repository (except `test.py`, which can be placed anywhere) has a header line detailing where in your Gym installation folder they should be placed. **`envs__init__.py` and `scoreboard__init__.py` both assume that you do not have any other custom gym environments installed!  If you have other custom gym environments, follow gym custom environment installation instructions found [here](https://github.com/openai/gym/wiki/Environments).**  

MAPR is not currently compatible with `gym_pull`, but I welcome pull requests that seek to resolve that issue.

Thanks to https://gist.github.com/vinothpandian/4337527 for the base code used for this gym environment.
