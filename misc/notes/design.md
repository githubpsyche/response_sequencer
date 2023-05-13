# Overview

The ResponseSequencer library is a Python library for research on automatic methods for preprocessing complex free response data as the production of a sequence of target items.

The project examines various frameworks that break down the task of coding free response data into discrete preprocessing steps and evaluates multiple automated techniques for performing each step independently and in the context of an overall preprocessing pipeline. Specifically, we compare human and computational methods for segmenting text, matching response units with target items, and identifying the sequence of target items generated in a trial.

The library provides and helps develop strategies for segmenting textual free response data into response units, matching response units with target items, and identifying the sequence of target items generated in a trial. The library also enables users to create and examine different evaluation implementations for assessing the performance of the preprocessing methods for their own datasets.

``` python
from response_sequencer import SentenceSegmenter, MaximumSimilarityMatcher, PipelineSequencer

# Initialize Segmenter and Matcher
segmenter = SentenceSegmenter()
matcher = MaximumSimilarityMatcher(language_model_string="sentence-transformers/paraphrase-distilroberta-base-v2")

# Create PipelineSequencer with Segmenter and Matcher
sequencer = PipelineSequencer(segmenter, matcher)

# Sample text
text = "OpenAI is an AI research lab founded by Elon Musk. GPT-4 is a large-scale language model developed by OpenAI. The Tesla Cybertruck is an electric vehicle created by Tesla Inc."

# List of target items (normally more complexly specified than this)
target_items = ["OpenAI", "GPT-4", "Tesla Cybertruck"]

# Sequence target items in the text
sequence = sequencer.sequence(text, target_items)

print(sequence)
# Output: ['OpenAI', 'GPT-4', 'Tesla Cybertruck']
```

# Architecture

The ResponseSequencer library follows a strategy pattern architecture with elements of the decorator pattern. It consists of multiple abstract classes of strategies, concrete classes for each strategy, and a decorator class to compose concrete classes. The architecture includes:

## Segmenter

The Segmenter abstract class defines the interface for implementations that segment raw response text into response units. Example concrete implementations of this class can use various approaches such as rule-based, machine learning, or natural language processing techniques.

### SentenceSegmenter

A baseline concrete class that applies `spacy`'s built-in sentence tokenizer to segment text into response units.

## Matcher

The Matcher abstract class defines the interface for implementations that match response units with target items. Example concrete implementations of this class can utilize methods such as keyword matching, similarity measures, or machine learning algorithms.

### MaximumSimilarityMatcher

A baseline concrete class that implements a maximum similarity approach for matching response units with target items. This Matcher utilizes cosine similarity and a sentence embedding loaded using the SentenceTransformers library to identify the response unit that is most similar to each target item as its best match.

## Sequencer

The Sequencer abstract class defines the interface for implementations that identify the sequence of target items generated in raw response text. Example concrete implementations of this class can be based on compositions of Segmenter and Matcher strategies or on more holistic heuristic or statistical methods.

### GPT4Sequencer

A concrete class that implements a GPT-4-based approach for identifying the sequence of target items generated in a trial. This sequencer utilizes the GPT-4 language model to predict the most likely sequence of target items recalled given the context of the response units.

### PipelineSequencer

A concrete class that allows users to compose concrete Segmenter and Matcher classes to identify the sequence of target items generated in a trial. This class should provide a flexible way to create and test various Segmenter and Matcher implementations combinations.

## Evaluation

Parallel abstract classes SegmeterEvaluator, MatcherEvaluator, and SequencerEvaluator define interfaces for implementing evaluation metrics for each strategy type. Concrete classes should provide methods for evaluating the performance of different implementations for Segmenter, Matcher, and Sequencer based on comparison to a gold standard dataset.

# Data

Data used for ongoing research contains the following arrays. Each row of an array corresponds to a specific trial. The structures `pres_itemids` and `pres_itemnos` are named that way for historical reasons but identify the pool of word senses available for recall in a given trial. Only a single stimulus is presented per trial.

| Structure Name    | Shape      | Data Type | Description                                                                                                                                                                                   | Example(s)                                                                                                                                                                                                                                                                                                                         |
|---------------|---------------|---------------|---------------|---------------|
| subject           | (2870, 1)  | int32     | unique identifiers ranging from 0-69 identifying the subject performing the trial                                                                                                             | 0                                                                                                                                                                                                                                                                                                                                  |
| subject_tag       | (2870, 1)  | int32     | original identifier used for each subject                                                                                                                                                     | 1005                                                                                                                                                                                                                                                                                                                               |
| stimulus          | (2870, 1)  | string    | The stimulus presented to the subject                                                                                                                                                         | 'ace'                                                                                                                                                                                                                                                                                                                              |
| stimulusid        | (2870, 1)  | int32     | A unique numeric identifier for each stimulus, 1-indexed                                                                                                                                      | 1                                                                                                                                                                                                                                                                                                                                  |
| condition         | (2870, 1)  | string    | unique identifier for group or condition under which the stimulus was presented                                                                                                               | 'NC', 'TBI'                                                                                                                                                                                                                                                                                                                        |
| listLength        | (2870, 1)  | int32     | The length of the presented list                                                                                                                                                              | 9                                                                                                                                                                                                                                                                                                                                  |
| pres_itemids      | (2870, 31) | int32     | The unique identifier for each sense available for recall in a given trial, with senses indexed across the full stimulus pool rather than between trial. 1-indexed, with 0 as the null value. | \[10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]                                                                                                                                                                                                                            |
| pres_itemnos      | (2870, 31) | int32     | Within-trial identifiers for each sense available for recall in a given trial. First sense always has an index of 1, with 0 as the null value                                                 | \[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]                                                                                                                                                                                                                                    |
| recalls           | (2870, 31) | int32     | pres_itemnos indices, arranged in the order in which they were recalled (if at all) in the trial. 0 as the null value                                                                         | \[3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]                                                                                                                                                                                                                                    |
| rec_itemids       | (2870, 31) | int32     | pres_itemids indices, arranged in the order in which they were recalled (if at all) in the trial                                                                                              | \[12, 11, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]                                                                                                                                                                                                                                 |
| response_units    | (2870, 31) | string    | Text of each utterance recalling a word sense; these units are arranged in the order that the corresponding senses were recalled. Empty strings '' as the null value.                         | \["I feel like there's also a meaning of wake that has something to do with water, but I can't think of what it is, exactly. Like, waves, maybe, something to do with, like, the waves. Um I don't know. I don't know.", "a wake can also be an, like, event, kind of like a funeral, or celebration of someone's life. Um, um" \] |
| recall_transcript | (2870, 1)  | string    | The unsegmented transcript of a trial, including timestamps and other information                                                                                                             | \[" Um, wake can be, like, you, it's, it can be like a verb, like ...\]                                                                                                                                                                                                                                                            |

A separate text file contains the text of each stimulus word. Each line in the file corresponds to the index of the stimulus in the primary dataset. For example, the first line of the file contains the text of the stimulus with index 1.

Another text file contains the definitions of each sense. Each line in the file corresponds to the index of the sense in the primary dataset. For example, the first line of the file contains the text of the definition of the sense with index 1.

Embeddings for each word and sense are stored in a pair of npy files. The first file contains the embeddings for each word, and the second file contains the embeddings for each sense. Each row in the file corresponds to the index of the word or sense in the primary dataset. For example, the file's first row contains the embedding for the word or sense with index 1.

In usage examples, the `hdf5storage` package loads structured data into Python. However, other formats for storing structured data should work as well.

# Usage

The following examples demonstrate how to use the ResponseSequencer library with concrete implementations.

## Loading Data

First, let's load the structured data using the `hdf5storage` package:

``` python
import hdf5storage
data = hdf5storage.loadmat('data.mat')

with open('sense_pool.txt', 'r') as f:
    sense_pool = f.read().split('\n')
```

## Creating Segmenter, Matcher, and Sequencer Instances

Next, instantiate concrete implementations for Segmenter, Matcher, and Sequencer:

``` python
from response_sequencer import SentenceSegmenter, MaximumSimilarityMatcher, GPT4Sequencer, PipelineSequencer

segmenter = SentenceSegmenter()

language_model_string = 'average_word_embeddings_glove.6B.300d'
matcher = MaximumSimilarityMatcher(language_model_string)

prompt = 'An idea unit is a meaningful text fragment that conveys a piece of the narrative. For example, in the sentence "One fine day an old Maine man was fishing on his favorite lake and catching very little.", there are four unique idea units: "One fine day", "an old Maine man", "was fishing on his favorite lake", and "and catching very little." The following story needs to be copied and segmented into idea units. Copy the following story word-for-word and start a new line whenever one idea unit ends and another begins. This is the story:'
sequencer = GPT4Sequencer(prompt)

pipeline_sequencer = PipelineSequencer(segmenter, matcher)
```

## Segmenting Text

Use the `SentenceSegmenter` instance to segment the response units:

``` python
response_text = data["recall_transcript"][0]
segmented_text = segmenter.segment(response_text)
```

## Matching Response Units with Target Items

Use the `MaximumSimilarityMatcher` instance to match response units with target items:

``` python
response_units = segmented_text
target_items = [senses_pool[each-1] for each in data["pres_itemids"][0] if each != 0]
matches = matcher.match(response_units, target_items)
```

## Identifying the Sequence of Target Items

Use the `GPT4Sequencer` or `PipelineSequencer` instances to identify the sequence of target items generated in a response text:

``` python
predicted_sequence_gpt4 = sequencer.sequence(response_text)
predicted_sequence_pipeline = pipeline_sequencer.sequence(response_text)
```

## Evaluating Implementations

Instantiate and use concrete evaluation classes to evaluate the performance of different implementations for Segmenter, Matcher, and Sequencer based on comparison to a gold standard dataset:

``` python
from response_sequencer.evaluation import SegmenterF1Evaluator, MatcherAccuracyEvaluator, SequencerAccuracyEvaluator

segmenter_evaluator = SegmenterF1Evaluator()
matcher_evaluator = MatcherAccuracyEvaluator()
sequencer_evaluator = SequencerAccuracyEvaluator()

# Assuming gold_standard_segmentation, gold_standard_matching, and gold_standard_sequencing are provided
segmenter_score = segmenter_evaluator.evaluate(segmented_text, gold_standard_segmentation)
matcher_score = matcher_evaluator.evaluate(matches, gold_standard_matching)
sequencer_score_gpt4 = sequencer_evaluator.evaluate(predicted_sequence_gpt4, gold_standard_sequencing)
sequencer_score_pipeline = sequencer_evaluator.evaluate(predicted_sequence_pipeline, gold_standard_sequencing)
```

These examples demonstrate the flexibility of the ResponseSequencer library, allowing users to create, examine, and evaluate various Segmenter, Matcher, Sequencer, and evaluation implementations based on their specific research needs.

# API Reference

## Segmenter

### `class response_sequencer.Segmenter`

Abstract base class for implementing text segmentation strategies. To create a custom segmenter, inherit from this class and override the `segment` method.

#### Methods

##### `segment(self, text: str) -> List[str]`

Segments the input text into response units.

-   **Parameters**
    -   `text` (str): The input text to be segmented.
-   **Returns**
    -   A list of response units (strings).

## SentenceSegmenter

## `class response_sequencer.SentenceSegmenter(Segmenter)`

Concrete implementation of `Segmenter` that uses `spacy`'s built-in sentence tokenizer to segment text into response units.

## Matcher

### `class response_sequencer.Matcher`

Abstract base class for implementing response unit matching strategies. To create a custom matcher, inherit from this class and override the `match` method.

#### Methods

##### `match(self, response_units: List[str], target_items: List[str]) -> List[Tuple[str, str]]`

Matches response units with target items.

-   **Parameters**
    -   `response_units` (List\[str\]): The list of response units to be matched.
    -   `target_items` (List\[str\]): The list of target items to be matched with response units.
-   **Returns**
    -   A list of tuples, where each tuple consists of a response unit (str) and its matched target item (str).

## MaximumSimilarityMatcher

### `class response_sequencer.MaximumSimilarityMatcher(Matcher)`

Concrete implementation of `Matcher` that uses a maximum similarity approach for matching response units with target items. This matcher utilizes cosine similarity and a sentence embedding loaded using the SentenceTransformers library.

#### Methods

##### `__init__(self, language_model_string: str)`

Initializes the `MaximumSimilarityMatcher`.

-   **Parameters**
    -   `language_model_string` (str): The string identifier for the sentence embedding model to use.

## Sequencer

### `class response_sequencer.Sequencer`

Abstract base class for implementing target item sequencing strategies. To create a custom sequencer, inherit from this class and override the `sequence` method.

#### Methods

##### `sequence(self, text: str) -> List[str]`

Identifies the sequence of target items generated in the input text.

-   **Parameters**
    -   `text` (str): The input text to be sequenced.
-   **Returns**
    -   A list of target items (strings) in their predicted sequence.

## GPT4Sequencer

### `class response_sequencer.GPT4Sequencer(Sequencer)`

Concrete implementation of `Sequencer` that uses a GPT-4-based approach for identifying the sequence of target items generated in a response text.

#### Methods

##### `__init__(self, prompt: str)`

Initializes the `GPT4Sequencer`.

-   **Parameters**
    -   `prompt` (str): The prompt used to guide the GPT-4 model for sequencing.

## PipelineSequencer

### `class response_sequencer.PipelineSequencer(Sequencer)`

Concrete implementation of `Sequencer` that composes `Segmenter` and `Matcher` classes to identify the sequence of target items generated in a response text.

#### Methods

##### `__init__(self, segmenter: Segmenter, matcher: Matcher)`

Initializes the `PipelineSequencer`.

-   **Parameters**
    -   `segmenter` (Segmenter): An instance of a concrete implementation of `Segmenter`.
    -   `matcher` (Matcher): An instance of a concrete implementation of `Matcher`.

##### `sequence(self, text: str, target_items: List[str]) -> List[str]`

Identifies the sequence of target items generated in the input text.

-   **Parameters**
    -   `text` (str): The input text to be sequenced.
    -   `target_items` (List\[str\]): The list of target items to be matched and sequenced.
-   **Returns**
    -   A list of target items (strings) in their predicted sequence.

## Evaluation

### `class response_sequencer.evaluation.SegmenterEvaluator`

Abstract class defining the interface for implementations of evaluation metrics for `Segmenter`.

#### Methods

##### `evaluate(self, predicted: List[str], gold_standard: List[str]) -> float`

Evaluates the performance of a `Segmenter` implementation based on comparison to a gold standard dataset.

-   **Parameters**
    -   `predicted` (List\[str\]): The list of predicted response units.
    -   `gold_standard` (List\[str\]): The list of gold standard response units.
-   **Returns**
    -   A float representing the evaluation score.

### `class response_sequencer.evaluation.MatcherEvaluator`

Abstract class defining the interface for implementations of evaluation metrics for `Matcher`.

#### Methods

##### `evaluate(self, predicted: List[str], gold_standard: List[str]) -> float`

Evaluates the performance of a `Matcher` implementation based on comparison to a gold standard dataset.

-   **Parameters**
    -   `predicted` (List\[str\]): The list of predicted matches.
    -   `gold_standard` (List\[str\]): The list of gold standard matches.
-   **Returns**
    -   A float representing the evaluation score.

### `class response_sequencer.evaluation.SequencerEvaluator`

Abstract class defining the interface for implementations of evaluation metrics for `Sequencer`.

#### Methods

##### `evaluate(self, predicted: List[str], gold_standard: List[str]) -> float`

Evaluates the performance of a `Sequencer` implementation based on comparison to a gold standard dataset.

-   **Parameters**
    -   `predicted` (List\[str\]): The list of predicted target item sequences.
    -   `gold_standard` (List\[str\]): The list of gold standard target item sequences.
-   **Returns**
    -   A float representing the evaluation score.

## Example Concrete Implementations

### `class response_sequencer.evaluation.SegmenterF1Evaluator(SegmenterEvaluator)`

Concrete implementation of `SegmenterEvaluator` that evaluates `Segmenter` performance using the F1 score.

### `class response_sequencer.evaluation.MatcherAccuracyEvaluator(MatcherEvaluator)`

Concrete implementation of `MatcherEvaluator` that evaluates `Matcher` performance using accuracy.

### `class response_sequencer.evaluation.SequencerAccuracyEvaluator(SequencerEvaluator)`

Concrete implementation of `SequencerEvaluator` that evaluates `Sequencer` performance using accuracy.