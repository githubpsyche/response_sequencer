# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/05_Sequencing.ipynb.

# %% auto 0
__all__ = ['Sequencer', 'PipelineSequencer']

# %% ../notebooks/05_Sequencing.ipynb 2
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class Sequencer(ABC):

    """
    Abstract base class for implementing target item sequencing strategies. To create a custom sequencer, inherit from this class and override the sequence method.
    """

    @abstractmethod
    def __call__(
        self, response_transcript: str, target_items: List[str], target_context: str = '') -> Dict[str, object]:
        """
        Identifies the sequence of target items in the input text using the
        provided Segmenter and Matcher instances.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            Dict[str, List[str]]: Dictionary containing:
                - 'target_context': The string containing the context of the target items. (if applicable)
                - 'target_items': The list of target items
                - 'response_transcript': The input text
                - 'response_units': The list of response units, a dictionary of the form {'text': str, 'span' [(start, end)]}
                - 'matches': a 2-D boolean numpy array of shape (len(target_items), len(response_units)) containing True if the target item matches the response unit at the corresponding index.
        """
        pass


# %% ../notebooks/05_Sequencing.ipynb 4
from .segmenting import Segmenter
from .matching import Matcher
import numpy as np

class PipelineSequencer(Sequencer):
    """
    Concrete implementation of Sequencer that composes Segmenter and Matcher
    classes to identify the sequence of target items generated in a response text.
    """

    def __init__(self, segmenter: Segmenter, matcher: Matcher):
        """
        Initializes the PipelineSequencer with a given Segmenter and Matcher.

        Parameters:
            segmenter (Segmenter): An instance of a concrete Segmenter implementation.
        matcher (Matcher): An instance of a concrete Matcher implementation.
        """
        self.segmenter = segmenter
        self.matcher = matcher

    def __call__(
        self, response_transcript: str, target_items: List[str], target_context: str = '') -> Dict[str, object]:
        """
        Identifies the sequence of target items in the input text using the
        provided Segmenter and Matcher instances.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            Dict[str, List[str]]: Dictionary containing:
                - 'target_context': The string containing the context of the target items. (if applicable)
                - 'target_items': The list of target items
                - 'response_transcript': The input text
                - 'response_units': The list of response units, a dictionary of the form {'text': str, 'span' [(start, end)]}
                - 'matches': a 2-D boolean numpy array of shape (len(target_items), len(response_units)) containing True if the target item matches the response unit at the corresponding index.
        """

        response_units = self.segmenter(response_transcript)
        matching = self.matcher(
            response_units, target_items, response_transcript, target_context)

        return {
            'target_context': target_context,
            'target_items': target_items,
            'response_transcript': response_transcript,
            'response_units': response_units,
            'matches': matching
        }
