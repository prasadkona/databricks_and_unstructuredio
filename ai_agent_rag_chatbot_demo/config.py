# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1&user_hash=223df685f11fc987eaee6acb09a24cf2155780c3263a4b2c44db6bb742132b2d">

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="one-env-shared-endpoint-6"

catalog = "prasad_kona_isv"

#email = spark.sql('select current_user() as user').collect()[0]['user']
#username = email.split('@')[0].replace('.', '_')
#dbName = db = f"dbdemos_rag_{username}"
dbName = db = "demo"

# COMMAND ----------

# MAGIC %md
# MAGIC ### License
# MAGIC This demo installs the following external libraries on top of DBR(ML):
# MAGIC
# MAGIC
# MAGIC | Library | License |
# MAGIC |---------|---------|
# MAGIC | langchain     | [MIT](https://github.com/langchain-ai/langchain/blob/master/LICENSE)     |
# MAGIC | lxml      | [BSD-3](https://pypi.org/project/lxml/)     |
# MAGIC | transformers      | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)     |
# MAGIC | unstructured      | [Apache 2.0](https://github.com/Unstructured-IO/unstructured/blob/main/LICENSE.md)     |
# MAGIC | llama-index      | [MIT](https://github.com/run-llama/llama_index/blob/main/LICENSE)     |
# MAGIC | tesseract      | [Apache 2.0](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)     |
# MAGIC | poppler-utils      | [MIT](https://github.com/skmetaly/poppler-utils/blob/master/LICENSE)     |
# MAGIC | textstat      | [MIT](https://pypi.org/project/textstat/)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC | evaluate      | [Apache2](https://pypi.org/project/evaluate/)     |
# MAGIC | torch      | [BDS-3](https://github.com/intel/torch/blob/master/LICENSE.md)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC
# MAGIC
# MAGIC
# MAGIC