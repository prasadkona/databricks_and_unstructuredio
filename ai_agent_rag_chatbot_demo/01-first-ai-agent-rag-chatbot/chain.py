# Databricks notebook source
# MAGIC %md
# MAGIC # Your langchain RAG chain
# MAGIC
# MAGIC In this notebook, we'll put together our langchain application.
# MAGIC
# MAGIC It's best practice to build your chain as a separate file and reference it directly to avoid serialization issues. 
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-first-step%2Fchain&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-first-step%2Fchain&version=1&user_hash=223df685f11fc987eaee6acb09a24cf2155780c3263a4b2c44db6bb742132b2d">

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.49.0 "databricks-langchain>=0.4.0" databricks-agents mlflow[databricks] databricks-vectorsearch==0.55 langchain==0.3.19 langchain_core==0.3.37 bs4==0.0.2 markdownify==0.14.1 pydantic==2.10.1
# MAGIC #%pip install openai
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from databricks_langchain import DatabricksEmbeddings

import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

## Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
    columns=["id","text","filename","type","text_as_html"],
    text_column="text",
    embedding = DatabricksEmbeddings(
                    endpoint=model_config.get("embedding_model"),
                )
).as_retriever(search_kwargs={"k": 3})

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
mlflow.models.set_retriever_schema(primary_key="id", text_column="text", doc_uri="filename")






from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain.chat_models import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("llm_prompt_template")), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)


# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)