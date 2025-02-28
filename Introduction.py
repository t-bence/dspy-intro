# Databricks notebook source
# MAGIC %md
# MAGIC Reference: https://dspy.ai/learn/

# COMMAND ----------

# MAGIC %pip install dspy

# COMMAND ----------

endpoint_name = "databricks-meta-llama-3-3-70b-instruct"

# COMMAND ----------

import dspy
lm = dspy.LM(f"databricks/{endpoint_name}", cache=False)
dspy.configure(lm=lm)

# COMMAND ----------

lm("Say this is a test!")

# COMMAND ----------

qa = dspy.ChainOfThought("question -> answer")

response = qa(question="How many floors are in the castle David Gregory inherited?", cache=False)

# COMMAND ----------

print(response.answer)
print(response.reasoning)

# COMMAND ----------

len(lm.history)
lm.history[0]

# COMMAND ----------

# MAGIC %md
# MAGIC # Signatures

# COMMAND ----------

sentence = "It's a charming and often affecting journey."
classify = dspy.Predict("sentence -> sentiment") # default argument is string
classify(sentence=sentence).sentiment

# COMMAND ----------

classify = dspy.Predict("sentence -> sentiment: bool")
classify(sentence=sentence).sentiment

# COMMAND ----------

document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)

print(response.keys())
print(response.reasoning)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Class-based signatures

# COMMAND ----------

from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)

# COMMAND ----------

# MAGIC %md
# MAGIC My custom example:

# COMMAND ----------

class Coder(dspy.Signature):
    """Write valid Python code."""

    task: str = dspy.InputField()
    code: str = dspy.OutputField()

task = "Write a fizz-buzz Python code."

programmer = dspy.Predict(Coder)
result = programmer(task=task)

# COMMAND ----------

def formatted(input: str) -> str:
    return input.encode().decode("unicode_escape")

print(formatted(result.code))

# COMMAND ----------

# MAGIC %md
# MAGIC # Program of Thought

# COMMAND ----------

from dspy import Signature

class BasicQA(Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ProgramOfThought Module
pot = dspy.ProgramOfThought(BasicQA)

# COMMAND ----------

result = pot(question="What is 1 + 1?")

# COMMAND ----------

print(result)

# COMMAND ----------

lm.history[-1]

# COMMAND ----------

lm.inspect_history(3)
