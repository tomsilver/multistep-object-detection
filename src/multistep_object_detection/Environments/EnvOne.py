"""Environment One: 1D grid word search environment.

This module provides a gymnasium environment for word search in a 1D character grid.
"""

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from multistep_object_detection.Environments.common_words import COMMON_WORDS_BY_LENGTH


class EnvOne(gym.Env):
    """A gymnasium environment for 1D word search tasks."""

    def __init__(
        self,
        size: int = 20,
        target_size: int = 3,
        alphabet=None,
        correct_reward=100,
        wrong_penalty=-50,
        step_penalty=-0.1,
        prop_empty=0.0,
        alphabet_real=False,
    ):
        super().__init__()

        # Set default alphabet if none provided
        if alphabet is None:
            alphabet = {
                "a": 1,
                "b": 1,
                "c": 1,
                "d": 1,
                "e": 1,
                "f": 1,
                "g": 1,
                "h": 1,
                "i": 1,
                "j": 1,
                "k": 1,
                "l": 1,
                "m": 1,
                "n": 1,
                "o": 1,
                "p": 1,
                "q": 1,
                "r": 1,
                "s": 1,
                "t": 1,
                "u": 1,
                "v": 1,
                "w": 1,
                "x": 1,
                "y": 1,
                "z": 1,
                "_": 1,
            }

        # These are the parameters for the grid that are defaulted above
        self.size = size
        self.target_size = target_size
        self.prop_empty = prop_empty
        self.alphabet_real = alphabet_real

        # Set up alphabet based on parameters
        if alphabet_real:
            # English letter frequencies (approximate) - completely override alphabet
            # These are the raw percentages, will be normalized below
            raw_frequencies = {
                "a": 8.12,
                "b": 1.49,
                "c": 2.78,
                "d": 4.25,
                "e": 12.02,
                "f": 2.23,
                "g": 2.02,
                "h": 6.09,
                "i": 6.97,
                "j": 0.15,
                "k": 0.77,
                "l": 4.03,
                "m": 2.41,
                "n": 6.75,
                "o": 7.51,
                "p": 1.93,
                "q": 0.10,
                "r": 5.99,
                "s": 6.33,
                "t": 9.06,
                "u": 2.76,
                "v": 0.98,
                "w": 2.36,
                "x": 0.15,
                "y": 1.97,
                "z": 0.07,
            }

            # Normalize frequencies to sum to (1 - prop_empty)
            total_freq = sum(raw_frequencies.values())
            target_sum = 1.0 - prop_empty

            self.alphabet = {}
            for char, freq in raw_frequencies.items():
                self.alphabet[char] = (freq / total_freq) * target_sum

            # Add empty spaces if needed
            if prop_empty > 0:
                self.alphabet["_"] = prop_empty

        else:
            # Use provided alphabet
            self.alphabet = alphabet.copy()

            # Adjust for empty spaces (using "_")
            if prop_empty > 0:
                # Normalize existing alphabet to sum to (1 - prop_empty)
                total_letters = sum(self.alphabet.values())
                target_sum = 1.0 - prop_empty

                # Scale down all existing frequencies
                for char in self.alphabet:
                    scaled_freq = self.alphabet[char] / total_letters
                    self.alphabet[char] = scaled_freq * target_sum

                # Add empty spaces
                self.alphabet["_"] = prop_empty

        # These are the rewards for the grid that are defaulted above
        self.correct_reward = correct_reward
        self.wrong_penalty = wrong_penalty
        self.step_penalty = step_penalty

        self._agent_location = -1  # Agent position
        self._grid: Optional[Any] = None  # The 1D grid of characters
        self._target_string: Optional[str] = None  # The target substring
        self._target_start_pos = -1  # Starting position of target in grid
        self._target_end_pos = -1  # Ending position of target in grid

        # For storing the guess coordinates when action=2 is chosen
        self._guess_start = 0
        self._guess_end = 0

        # Action space: 0=left, 1=right, 2=guess
        self.action_space = spaces.Discrete(3)

        # This is the box space for when the agent guesses
        self.guess_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.int32  # [start_pos, end_pos]
        )

        # Observation space: dictionary
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Discrete(size),  # Current position
                # 0 is error, 1-26 is a-z, 27 is _
                "current_character": spaces.Discrete(28),
                "target_string": spaces.Text(max_length=target_size),  # Target string
            }
        )

    def _generate_grid(self) -> Tuple[Any, str, int, int]:
        """Generate a grid based on probabilities, size, target_string, and target_size.

        This uses np.random.choice, a great function that can basically do the request
        function automatically. Uses the environment's seeded random number generators
        for reproducibility.
        """
        # Getting characters and probabilities
        chars = list(self.alphabet.keys())
        probs = list(self.alphabet.values())

        # Normalizing the probabilities
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        # Generate random grid using seeded numpy random generator
        grid = self.np_random.choice(chars, size=self.size, p=probs)

        # Choose a random target string from common words list using seeded random
        if self.target_size in COMMON_WORDS_BY_LENGTH:
            target_string = self.np_random.choice(
                COMMON_WORDS_BY_LENGTH[self.target_size]
            )
        else:
            # Fallback: choose a random substring from the grid if target_size
            # not in common words
            target_start = self.np_random.integers(0, self.size - self.target_size + 1)
            target_string = "".join(
                grid[target_start : target_start + self.target_size]
            )
            return (
                grid,
                target_string,
                int(target_start),
                int(target_start + self.target_size - 1),
            )

        # Randomly decide if target should be forwards or backwards
        is_backwards = self.np_random.choice([True, False])
        if is_backwards:
            target_string = target_string[::-1]  # Reverse the string

        # Place the target string at a random position using seeded random
        target_start = self.np_random.integers(0, self.size - self.target_size + 1)
        for i, char in enumerate(target_string):
            grid[target_start + i] = char

        return (
            grid,
            target_string,
            int(target_start),
            int(target_start + self.target_size - 1),
        )

    def _char_to_num(self, char: str) -> int:
        """Convert character to number for observation space.

        Returns 0 if some error has occurred, 1-26 for letters a through z, and 27 for
        "_" or blank.
        """
        if char == "_":
            return 27
        if "a" <= char <= "z":
            return ord(char) - ord("a") + 1
        return 0

    def _get_obs(self) -> dict:
        """Get current observation."""
        if self._grid is not None and 0 <= self._agent_location < self.size:
            current_char = self._grid[self._agent_location]
        else:
            current_char = "_"

        return {
            "agent_location": self._agent_location,
            "current_character": self._char_to_num(current_char),
            "target_string": self._target_string,
        }

    def _get_info(self) -> dict:
        """Get auxiliary debugging information."""
        grid_str = "".join(self._grid) if self._grid is not None else ""
        return {
            "target_start_pos": self._target_start_pos,
            "target_end_pos": self._target_end_pos,
            "grid": grid_str,
        }

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """Start a new episode."""
        super().reset(**kwargs)

        # Generate new grid and target
        (
            self._grid,
            self._target_string,
            self._target_start_pos,
            self._target_end_pos,
        ) = self._generate_grid()

        # Place agent at middle
        self._agent_location = self.size // 2

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Execute one timestep."""
        terminated = False
        truncated = False
        reward = self.step_penalty  # Default penalty for all actions

        if action == 0:  # Move left
            if self._agent_location > 0:
                self._agent_location -= 1
        elif action == 1:  # Move right
            if self._agent_location < self.size - 1:
                self._agent_location += 1
        elif action == 2:  # Guess
            # Check if guess is correct
            if (
                self._guess_start == self._target_start_pos
                and self._guess_end == self._target_end_pos
            ):
                reward = self.correct_reward
                terminated = True  # End program if right
            else:
                reward = self.wrong_penalty  # Guess error if wrong

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def set_guess(self, start_pos: int, end_pos: int):
        """Set the guess coordinates before calling step(2)."""
        self._guess_start = start_pos
        self._guess_end = end_pos

    def make_guess(self, start_pos: int, end_pos: int):
        """Alternative method: set guess and execute in one call."""
        self.set_guess(start_pos, end_pos)
        return self.step(2)

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            grid_str = "".join(self._grid) if self._grid is not None else ""
            print(f"\nGrid: {grid_str}")
            print(f"Agent: {' ' * self._agent_location}^")
            target_pos = f"{self._target_start_pos}-{self._target_end_pos}"
            print(f"Target: '{self._target_string}' at positions {target_pos}")
            if self._grid is not None and 0 <= self._agent_location < self.size:
                print(f"Current character: '{self._grid[self._agent_location]}'")
            else:
                print("Current character: '_'")

    def close(self):
        """Clean up."""
