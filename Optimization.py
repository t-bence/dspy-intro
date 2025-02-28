# Databricks notebook source
# MAGIC %pip install dspy datasets

# COMMAND ----------

#endpoint_name = "databricks-meta-llama-3-3-70b-instruct"
endpoint_name = "openai-4o-instruct"

import dspy
lm = dspy.LM(f"databricks/{endpoint_name}", cache=False, max_tokens=2000)
dspy.configure(lm=lm)

# COMMAND ----------

from datasets import load_dataset

imdb = load_dataset("stanfordnlp/imdb")

# COMMAND ----------

train = imdb["train"][:50]

# COMMAND ----------

# MAGIC %md
# MAGIC Set up the base object itself

# COMMAND ----------

from typing import Literal

class SentimentAnalysis(dspy.Signature):
    """Extract the sentiment from the movie review."""

    review: str = dspy.InputField(desc="The movie review")
    sentiment: Literal[0, 1] = dspy.OutputField(desc="The sentiment of the review, 0 for negative, 1 for positive")

# COMMAND ----------

sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysis)

# COMMAND ----------

# MAGIC %md
# MAGIC Create the Examples

# COMMAND ----------

def prepare_examples(dataset: list, number: int) -> list[dspy.Example]:
    result = []
    for i in range(number):
        row = dataset[i]
        result.append(
            dspy.Example(
                review=row["text"],
                sentiment=row["label"]
            ).with_inputs("review")
        )
    return result


# COMMAND ----------

train_set = prepare_examples(imdb["train"], 10)
test_set = prepare_examples(imdb["test"], 10)

# COMMAND ----------

# MAGIC %md
# MAGIC Define the metric function

# COMMAND ----------

def correct_sentiment_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
  return prediction.sentiment == example.sentiment

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate

# COMMAND ----------

evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=correct_sentiment_metric,
    num_threads=4,
    display_progress=True,
    display_table=True
)

# COMMAND ----------

evaluate_correctness(sentiment_analyzer)

# COMMAND ----------

# MAGIC %md
# MAGIC Optimize the program

# COMMAND ----------

mipro_optimizer = dspy.MIPROv2(
    metric=correct_sentiment_metric,
    auto="medium",
)
optimized_sentiment_analyzer = mipro_optimizer.compile(
    sentiment_analyzer,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False
)

# COMMAND ----------

dspy.inspect_history(n=1)

# COMMAND ----------

cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # cost in USD, as calculated by LiteLLM for certain providers
cost

# COMMAND ----------

evaluate_correctness(optimized_sentiment_analyzer)

# COMMAND ----------

optimized_sentiment_analyzer.save("optimized_sentiment_analyzer.json")

# COMMAND ----------

optimized_sentiment_analyzer(review=test_set[0].review)

# COMMAND ----------

sentiment_analyzer.save("original.json")
