# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/01_Segmenting.ipynb.

# %% auto 0
__all__ = ['Segmenter', 'SentenceSegmenter', 'AllSentenceFragmentsSegmenter', 'MultiSentenceFragmentsSegmenter',
           'AllFragmentsSegmenter', 'ClausiePropositionSegmenter', 'map_segments_to_original_text',
           'SimpleSentenceSegmenter']

# %% ../notebooks/01_Segmenting.ipynb 2
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

class Segmenter(ABC):
    """
    Abstract base class for implementing text segmentation strategies.
    """

    @abstractmethod
    def __call__(self, text: str) -> List[Dict[str, object]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        pass

# %% ../notebooks/01_Segmenting.ipynb 4
import spacy

class SentenceSegmenter(Segmenter):

    """
    Concrete Segmenter class that identifies the sentences in the input text using the SpaCy library.
    """

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")

    def __call__(self, text: str) -> List[Dict[str, object]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        return [{"text": sent.text, "spans": [(sent.start_char, sent.end_char)]} for sent in self._nlp(text).sents]

# %% ../notebooks/01_Segmenting.ipynb 7
import spacy
from typing import List
from spacy.tokens import Span

class AllSentenceFragmentsSegmenter(Segmenter):

    """
    Concrete Segmenter class that generates all possible segments of tokens in each sentence.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, text: str) -> List[List[Tuple[str, Tuple[int, int]]]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        segments = []
        for sentence in self.nlp(text).sents:
            for i in range(len(sentence)):
                for j in range(i + 1, len(sentence) + 1):

                    sentence_fragment = sentence[i:j]
                    if sentence_fragment.text.strip():
                        segments.append({
                            "text": sentence_fragment.text,
                            "spans": [(sentence_fragment.start_char, sentence_fragment.end_char)]
                        })

        return segments

# %% ../notebooks/01_Segmenting.ipynb 10
import spacy
from typing import List, Tuple
from spacy.tokens import Doc

class MultiSentenceFragmentsSegmenter(Segmenter):

    """
    Concrete Segmenter class that generates all possible segments of tokens
    within a specified number of sentences in the input text.
    """

    def __init__(self, max_sentences: int, min_tokens: int = 1):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.Defaults.stop_words.add("um")
        self.nlp.Defaults.stop_words.add("uh")

        self.max_sentences = max_sentences
        self.min_tokens = min_tokens

    def __call__(self, text: str) -> List[Dict[str, object]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        segments = []
        doc = self.nlp(text)
        sentences = list(doc.sents)
        num_sentences = len(sentences)

        for start_idx in range(num_sentences):
            for end_idx in range(start_idx + 1, min(start_idx + self.max_sentences + 1, num_sentences + 1)):
                segment = doc[sentences[start_idx].start:sentences[end_idx - 1].end]

                for i in range(len(segment)):
                    for j in range(i + 1, len(segment) + 1):

                        fragment = segment[i:j]

                        token_count = sum(not (token.is_punct or token.is_stop) for token in fragment)

                        if token_count >= self.min_tokens and fragment.text.strip():
                            segments.append({
                                "text": fragment.text,
                                "spans": [(fragment.start_char, fragment.end_char)]
                            })

        return segments


# %% ../notebooks/01_Segmenting.ipynb 13
import spacy
from typing import List, Tuple

class AllFragmentsSegmenter:
    """
    Concrete Segmenter class that generates all possible segments of tokens in the entire text.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Tuple[str, Tuple[int, int]]]: The list of segmented response units.
            
            Each unit is a tuple with the following elements:
            - The text representation of the fragment.
            - A tuple representing the character-level start and end indices of the fragment in the input text.
        """
        segments = []
        doc = self.nlp(text)

        # Iterate over all possible fragments of tokens in the entire text
        for i in range(len(doc)):
            for j in range(i + 1, len(doc) + 1):
                fragment = doc[i:j]
                if fragment.text.strip():
                    segments.append({'text': fragment.text, 'spans':[(fragment.start_char, fragment.end_char)]})

        return segments

# %% ../notebooks/01_Segmenting.ipynb 16
import spacy
import claucy
from typing import List


class ClausiePropositionSegmenter(Segmenter):

    """
    Concrete Segmenter class that generates propositions from sentences using spacy-clausie.
    """

    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm")
        claucy.add_to_pipe(self._nlp)

    def __call__(self, text: str) -> List[List[Tuple[str, Tuple[int, int]]]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        propositions = []

        for sent in self._nlp(text).sents:
            for clause in sent._.clauses:
                for prop in clause.to_propositions(inflect=None):
                    propositions.append({
                        "text": " ".join([p.text for p in prop]),
                        "spans": [(p.start_char, p.end_char) for p in prop]
                    })

        return propositions

# %% ../notebooks/01_Segmenting.ipynb 19
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from spacy.tokens.span import Span
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

def map_segments_to_original_text(
        original_text: str, segments: List[str], start: int,
        model_name: str = "sentence-transformers/paraphrase-distilroberta-base-v1") -> List[Tuple[int, int]]:
    """
    Uses maximum similarity matching to map the segment to a span the original sentence.

    Parameters:
        original_text (str): The original sentence.
        segments (List[Span]): The list of segments of the original sentence.
        model_name (str): The name of the sentence embedding model.

    Returns:
        List[Tuple[int, int]]: The list of character-level start and end indices (tuples) in the original sentence.
    """
    
    model = SentenceTransformer(model_name)
    nlp = spacy.load("en_core_web_sm")
    original_doc = nlp(original_text)

    all_segments = []
    for i in range(len(original_text)):
        for j in range(i + 1, len(original_doc) + 1):
            all_segments.append(original_doc[i:j])

    segment_embeddings = np.array(model.encode([s.text for s in all_segments]))
    simplified_sentence_embeddings = np.array(model.encode(segments))
    similarity_matrix = cosine_similarity(simplified_sentence_embeddings, segment_embeddings)

    # Find the most similar segment for each simplified sentence
    most_similar_segment_indices = np.argmax(similarity_matrix, axis=1)
    simplified_to_original_mapping = [all_segments[index] for index in most_similar_segment_indices]

    return [(s.start_char + start, s.end_char + start) for s in simplified_to_original_mapping]

class SimpleSentenceSegmenter(Segmenter):
    """
    Implementation of the Segmenter abstract base class that recursively segments text into sentences using spacy,
    and generates the simplest possible sentences using the ctrl44-simp model.
    """

    def __init__(self, model_name: str = "liamcripwell/ctrl44-simp", depth_limit=1, sep_type="<ssplit>"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.depth_limit = depth_limit
        self.sep_type = sep_type

    def __call__(self, text: str) -> List[Dict[str, object]]:
        """
        Splits the input text into a list of response units.

        Parameters:
            text (str): The input text to be segmented.

        Returns:
            List[Dict[str, object]]: The list of segmented response units.
            
            Each unit is a dictionary with keys "text" and "spans":
            The "text" key corresponds to the text representation, 
            The "spans" key corresponds to a list of character-level start and end indices (tuples) in the input text.
        """
        segments = []
        for sentence in self.nlp(text).sents:
            simplified_sentences = self._simplify_sentence_recursive(sentence, depth=1)
            spans = map_segments_to_original_text(sentence.text, [s.text for s in simplified_sentences], start=sentence.start_char)

            segments.extend([{"text": s.text, "spans": [spans[i]]} for i, s in enumerate(simplified_sentences)])

        return segments

    def _simplify_sentence_recursive(self, sentence: Span, depth) -> List[Span]:
        """
        Recursively simplifies the input sentence using the specfied model.

        Parameters:
            sentence (Span): The input sentence to be simplified.
            depth (int): The current depth of the recursion.

        Returns:
            List[Span]: The list of simplified sentences.
        """

        # probes model to produce a string containing at least one sentence
        inputs = self.tokenizer(self.sep_type + sentence.text, return_tensors="pt")
        outputs = self.model.generate(**inputs, num_beams=10, max_length=128)
        output_string = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # splits the output string into sentences
        generated_sentences = [sent for sent in self.nlp(output_string).sents if sent.text.upper().isupper()]

        # if the output string contains only one sentence, or the sentence is unchanged, return the sentence
        if len(generated_sentences) == 1 or sentence.text in [sent.text for sent in generated_sentences]:
            return [sentence]
        elif depth == self.depth_limit:
            return generated_sentences
        else:
            simplest_sentences = []
            for generated_sentence in generated_sentences:
                simplest_sentences.extend(self._simplify_sentence_recursive(generated_sentence, depth=depth+1))
            return simplest_sentences