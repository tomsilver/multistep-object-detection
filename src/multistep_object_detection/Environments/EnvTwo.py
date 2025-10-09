"""EnvTwo module containing a 2D grid word search gymnasium environment.

This module implements a 2D grid-based word search environment where an agent can move
in four directions and make guesses about target word locations.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common_words import COMMON_WORDS_BY_LENGTH


class EnvTwo(gym.Env):
    """2D grid word search environment.

    A gymnasium environment where an agent navigates a 2D grid to find target words. The
    agent can move in four directions and make guesses about word locations.
    """

    def __init__(self, size: int = 15, target_size: int = 3, alphabet=None,
                 correct_reward=100, wrong_penalty=-50, step_penalty=-0.1,
                 prop_empty=0.0, alphabet_real=False, allow_backwards=True):
        """Initialize the 2D word search environment."""
        super().__init__()

        # Set default alphabet if none provided
        if alphabet is None:
            alphabet = {
                "a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1, "g": 1, "h": 1,
                "i": 1, "j": 1, "k": 1, "l": 1, "m": 1, "n": 1, "o": 1, "p": 1,
                "q": 1, "r": 1, "s": 1, "t": 1, "u": 1, "v": 1, "w": 1, "x": 1,
                "y": 1, "z": 1, "_": 1
            }

        # These are the parameters for the grid that are defaulted above
        self.size = size  # n x n grid
        self.target_size = target_size
        self.prop_empty = prop_empty
        self.alphabet_real = alphabet_real
        self.allow_backwards = allow_backwards

        # Set up alphabet based on parameters
        if alphabet_real:
            # English letter frequencies (approximate) - completely override alphabet
            # These are the raw percentages, will be normalized below
            raw_frequencies = {
                "a": 8.12, "b": 1.49, "c": 2.78, "d": 4.25, "e": 12.02,
                "f": 2.23, "g": 2.02, "h": 6.09, "i": 6.97, "j": 0.15,
                "k": 0.77, "l": 4.03, "m": 2.41, "n": 6.75, "o": 7.51,
                "p": 1.93, "q": 0.10, "r": 5.99, "s": 6.33, "t": 9.06,
                "u": 2.76, "v": 0.98, "w": 2.36, "x": 0.15, "y": 1.97,
                "z": 0.07
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
                    self.alphabet[char] = (
                        (self.alphabet[char] / total_letters) * target_sum
                    )

                # Add empty spaces
                self.alphabet["_"] = prop_empty

        # These are the rewards for the grid that are defaulted above
        self.correct_reward = correct_reward
        self.wrong_penalty = wrong_penalty
        self.step_penalty = step_penalty

        self._agent_location = [-1, -1]  # Agent position [row, col]
        self._grid = None  # The 2D grid of characters
        self._target_string = None  # The target substring
        self._target_start_pos = [-1, -1]  # Starting position of target [row, col]
        self._target_end_pos = [-1, -1]  # Ending position of target [row, col]
        self._target_direction = None  # Direction: 'horizontal' or 'vertical'
        self._target_orientation = None  # Orientation: 'forward' or 'backward'

        # For storing the guess coordinates when action=4 is chosen
        self._guess_start = [0, 0]
        self._guess_end = [0, 0]

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=guess
        self.action_space = spaces.Discrete(5)

        # This is the box space for when the agent guesses
        self.guess_space = spaces.Box(
            low=0,
            high=size-1,
            shape=(4,),  # [start_row, start_col, end_row, end_col]
            dtype=np.int32
        )

        # Observation space: dictionary
        self.observation_space = spaces.Dict({
            "agent_location": spaces.Box(
                low=0, high=size-1, shape=(2,), dtype=np.int32
            ),  # [row, col]
            "current_character": spaces.Discrete(29),  # 1-26 is a-z, 27 is _
            "target_string": spaces.Text(max_length=target_size),  # The target string
        })

    def _generate_grid(self):
        """Generating a 2D grid based on the probabilities, size, target_string, and
        target_size.

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

        # Generate random 2D grid using seeded numpy random generator
        grid = self.np_random.choice(chars, size=(self.size, self.size), p=probs)

        # Choose a random target string from common words list using seeded random
        if self.target_size not in COMMON_WORDS_BY_LENGTH:
            # Fallback: choose random substring if target_size not in common words
            row = self.np_random.integers(0, self.size)
            col = self.np_random.integers(0, self.size - self.target_size + 1)
            target_string = ''.join(grid[row, col:col + self.target_size])
            return (
                grid, target_string, [row, col],
                [row, col + self.target_size - 1], 'horizontal', 'forward'
            )

        target_string = self.np_random.choice(
            COMMON_WORDS_BY_LENGTH[self.target_size]
        )

        # Randomly choose direction (horizontal or vertical)
        direction = self.np_random.choice(['horizontal', 'vertical'])

        # Randomly choose orientation (forward or backward)
        orientation = 'forward'
        if self.allow_backwards:
            orientation = self.np_random.choice(['forward', 'backward'])

        # If backward, reverse the target string
        if orientation == 'backward':
            target_string = target_string[::-1]

        # Place the target string at a random position in the grid
        if direction == 'horizontal':
            # Horizontal placement
            row = self.np_random.integers(0, self.size)
            col = self.np_random.integers(0, self.size - self.target_size + 1)

            for i, char in enumerate(target_string):
                grid[row, col + i] = char

            target_start = [row, col]
            target_end = [row, col + self.target_size - 1]
        else:
            # Vertical placement
            row = self.np_random.integers(0, self.size - self.target_size + 1)
            col = self.np_random.integers(0, self.size)

            for i, char in enumerate(target_string):
                grid[row + i, col] = char

            target_start = [row, col]
            target_end = [row + self.target_size - 1, col]

        return grid, target_string, target_start, target_end, direction, orientation

    def _char_to_num(self, char):
        """This is solely for the observation space, printing out 0 if some error has
        occurred, 1-26 for all the letters a through z, and 27 for if the spot is "_" or
        blank."""
        if char == '_':
            return 27
        if char == '#':
            return 28
        if 'a' <= char <= 'z':
            return ord(char) - ord('a') + 1
        return 0

    def _get_obs(self):
        """Get current observation including agent location, current character, and
        target string."""
        row, col = self._agent_location
        if 0 <= row < self.size and 0 <= col < self.size:
            current_char = self._grid[row, col]
        else:
            current_char = '_'

        return {
            "agent_location": np.array(self._agent_location, dtype=np.int32),
            "current_character": self._char_to_num(current_char),
            "target_string": self._target_string,
        }

    def _get_info(self):
        """This is auxiliary debugging information."""
        return {
            "target_start_pos": self._target_start_pos,
            "target_end_pos": self._target_end_pos,
            "target_direction": self._target_direction,
            "target_orientation": self._target_orientation,
            "grid": self._grid.tolist(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment for a new episode.

        Generates new random 2D environment, places agent at center position, and
        returns initial observation
        """
        super().reset(seed=seed, options=options)

        # Generate new grid and target
        (self._grid, self._target_string, self._target_start_pos,
         self._target_end_pos, self._target_direction,
         self._target_orientation) = self._generate_grid()

        # Place agent at center
        self._agent_location = [self.size // 2, self.size // 2]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep in the environment.

        Actions: 0=up, 1=down, 2=left, 3=right, 4=guess
        Returns observation (current character), reward, termination status
        Episode ends when target is found or max steps reached
        """
        terminated = False
        truncated = False
        reward = 0.0  # Initialize reward to avoid possibly-used-before-assignment

        if action == 0:  # Move up
            if self._agent_location[0] > 0:
                self._agent_location[0] -= 1
            reward = self.step_penalty  # Step error (penalty)

        elif action == 1:  # Move down
            if self._agent_location[0] < self.size - 1:
                self._agent_location[0] += 1
            reward = self.step_penalty  # Step error (penalty)

        elif action == 2:  # Move left
            if self._agent_location[1] > 0:
                self._agent_location[1] -= 1
            reward = self.step_penalty  # Step error (penalty)

        elif action == 3:  # Move right
            if self._agent_location[1] < self.size - 1:
                self._agent_location[1] += 1
            reward = self.step_penalty  # Step error (penalty)

        elif action == 4:  # Guess
            # Check if guess is correct
            if (self._guess_start == self._target_start_pos and
                    self._guess_end == self._target_end_pos):
                reward = self.correct_reward
                terminated = True  # End program if right
            else:
                reward = self.wrong_penalty  # Guess error if wrong

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def set_guess(self, start_row: int, start_col: int, end_row: int, end_col: int):
        """Set the guess coordinates before calling step(4)"""
        self._guess_start = [start_row, start_col]
        self._guess_end = [end_row, end_col]

    def make_guess(self, start_row: int, start_col: int, end_row: int, end_col: int):
        """Alternative method: set guess and execute in one call"""
        self.set_guess(start_row, start_col, end_row, end_col)
        return self.step(4)

    def render(self, mode='human'):
        """Render the current state of the environment.

        Displays: 2D grid, agent position, target string location,
        and target details for debugging
        """
        if mode == 'human':
            print(f"\n2D Grid ({self.size}x{self.size}):")

            # Print grid as is
            for row in range(self.size):
                row_str = ""
                for col in range(self.size):
                    row_str += self._grid[row, col]
                print(row_str)

            print(f"Agent at: ({self._agent_location[0]}, {self._agent_location[1]})")
            target_info = (
                f"Target: '{self._target_string}' "
                f"({self._target_direction}, {self._target_orientation})"
            )
            print(target_info)
            location_info = (
                f"Target location: ({self._target_start_pos[0]}, "
                f"{self._target_start_pos[1]}) to ({self._target_end_pos[0]}, "
                f"{self._target_end_pos[1]})"
            )
            print(location_info)

            # Show current character
            row, col = self._agent_location
            if 0 <= row < self.size and 0 <= col < self.size:
                current_char = self._grid[row, col]
                print(f"Current character: '{current_char}'")

    def close(self):
        """Clean up resources when environment is closed."""
