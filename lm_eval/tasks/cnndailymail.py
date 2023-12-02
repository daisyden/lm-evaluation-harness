# TODO: Remove all TODO comments once the implementation is complete.
"""
Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond
https://aclanthology.org/K16-1028.pdf

The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.

Homepage: https://huggingface.co/datasets/cnn_dailymail
"""

from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import regex
import string
import evaluate
import numpy as np
import nltk

nltk.download('punkt')


_CITATION = """
@inproceedings{see-etal-2017-get,
    title = "Get To The Point: Summarization with Pointer-Generator Networks",
    author = "See, Abigail  and
      Liu, Peter J.  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-1099",
    doi = "10.18653/v1/P17-1099",
    pages = "1073--1083",
    abstract = "Neural sequence-to-sequence models have provided a viable new approach for abstractive text summarization (meaning they are not restricted to simply selecting and rearranging passages from the original text). However, these models have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways. First, we use a hybrid pointer-generator network that can copy words from the source text via pointing, which aids accurate reproduction of information, while retaining the ability to produce novel words through the generator. Second, we use coverage to keep track of what has been summarized, which discourages repetition. We apply our model to the CNN / Daily Mail summarization task, outperforming the current abstractive state-of-the-art by at least 2 ROUGE points.",
}
"""

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

instruction_template = "Summarize the following news article:"
text_column = "article"
targets = "highlights" 


class CNNDailyMail(Task):
    VERSION = None  
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "cnn_dailymail"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "3.0.0" 

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False 

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True 

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False 

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]


    def _process_doc(self, doc):
        # Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.

        x = {}
        x["instruction"] = instruction_template
        x["input"] = doc[text_column]
        x["output"] = doc[targets]

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        source = prompt_input.format_map(x)
        target = doc[targets] 

        return { "text": source, "target": target } 

    def doc_to_text(self, doc):
        # Format the query prompt portion of the document example.
        return doc["text"] 

    def doc_to_target(self, doc):
        # Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.

        return " " + doc["target"]

    def _normalize_answer(self, text):
        # Lowercase and remove punctuation, strip whitespace
        text = text.strip().lower().translate(str.maketrans("", "", string.punctuation))

        # Remove articles, resulting in duplicate whitespace
        text = regex.sub(r"\b(a|an|the)\b", " ", text)

        # Remove duplicate whitespace
        text = " ".join(text.split())

        return text

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # Construct your language model requests with the request factory, `rf`,
        # and return them as an iterable.
        continuation = rf.greedy_until(ctx, {"until": ["</s>"]})
        return continuation 

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        continuation = results[0].strip()

        continuation = "\n".join(nltk.sent_tokenize(continuation))

        #continuation = self._normalize_answer(results[0])
        #answers = [self._normalize_answer(answer) for answer in doc["highlights"]]
        #import pdb
        #pdb.set_trace()

        answer = doc["target"].strip()
        answer = "\n".join(nltk.sent_tokenize(answer))

        metric = evaluate.load("rouge")
        result = metric.compute(predictions=[continuation], references=[answer], use_stemmer=True, use_aggregator=False) 
        result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}

        return result 

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {
                "rouge1": mean,
                "rouge2": mean,
                "rougeL": mean,
                "rougeLsum": mean,
                }

    def higher_is_better(self):
        # For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {
                "rouge1": mean,
                "rouge2": mean,
                "rougeL": mean,
                "rougeLsum": mean,
                }

