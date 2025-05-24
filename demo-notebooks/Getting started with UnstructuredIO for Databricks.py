# Databricks notebook source
# MAGIC %pip  install httpx "unstructured[dropbox]"==0.12.6 "unstructured[pdf]"==0.12.6  "unstructured[databricks-volumes]"==0.12.6
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC UnstructuredIO provides powerfun capabilities to extract data from unstructured datasources and files. 
# MAGIC * They provide source conenctors to sources like gdrive, sharepoint, dropbox, box and others. 
# MAGIC * They extract text from files using various formats like pdf, html, sheets, slides, documents, excel, powerpoint, word and others

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In this notebook
# MAGIC * Dropbox contains a few pdf files under a folder called 'databricks-pdf'
# MAGIC * UnstructuredIO is used to extract the files from dropbox
# MAGIC * UnstructuredIO then for each of the files extracts the text
# MAGIC * UnstructuredIO then writes the extracted text as json to an Databricks UC Volume
# MAGIC
# MAGIC In this notebook we illustrate 4 scenarios
# MAGIC * 1. Use unstructuredIO oss to extract files from dropbox and land them to uc volumes. Extract text from pdf's and write the output json to UC volumes
# MAGIC * 2. Use unstructuredIO SAAS api to extract files from dropbox and land them to uc volumes. Extract text from pdf's and write the output json to UC volumes
# MAGIC * 3. Use unstructuredIO oss to read a file on uc volumes. Extract text from pdf's and write the output json to UC volumes
# MAGIC * 4. Use unstructuredIO SAAS api to read a file on uc volumes. Extract text from pdf's and write the output json to UC volumes
# MAGIC

# COMMAND ----------

# DBTITLE 1,Initialize Variables
# initialize variables
DROPBOX_TOKEN = dbutils.secrets.get("databricks_secret_scope_name", "DROPBOX_API_TOKEN")
UNSTRUCTURED_API_KEY = dbutils.secrets.get("databricks_secret_scope_name", "UNSTRUCTURED_API_KEY")
UNSTRUCTURED_SAAS_BASE_URL = dbutils.secrets.get("prasad_kona", "UNSTRUCTURED_SAAS_BASE_URL")

# url for the databricks workspace
databricks_host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

DATABRICKS_TOKEN = dbutils.secrets.get("databricks_secret_scope_name", "databricks_user_token")
DATABRICKS_HOSTNAME = dbutils.secrets.get("databricks_secret_scope_name", "databricks_hostname")
DATABRICKS_USERNAME = dbutils.secrets.get("databricks_secret_scope_name", "databricks_user_name")

catalog_name ="demo_catalog"
schema_name = "unstructured_io_demo_schema"
volume_name = "unstructured_io_demo_volume"

output_dir_name = f"/Users/{DATABRICKS_USERNAME}/dropbox-output"

download_dir_name = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/unstructured_io/raw"
processed_output_volume_path =f"{volume_name}/unstructured_io"
processed_output_volume_full_path = f"/Volumes/{catalog_name}/{schema_name}/{processed_output_volume_path}/structured-output"


# COMMAND ----------

# DBTITLE 1,1. Using UnstructuredIO oss source with Databricks
import os

from unstructured.ingest.connector.fsspec.dropbox import DropboxAccessConfig, SimpleDropboxConfig
from unstructured.ingest.interfaces import (
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import DropboxRunner


from unstructured.ingest.connector.databricks_volumes import (
    DatabricksVolumesAccessConfig,
    DatabricksVolumesWriteConfig,
    SimpleDatabricksVolumesConfig,
)
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import (
    ChunkingConfig,
    EmbeddingConfig,
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import LocalRunner
from unstructured.ingest.runner.writers.base_writer import Writer
from unstructured.ingest.runner.writers.databricks_volumes import (
    DatabricksVolumesWriter,
)



def get_writer() -> Writer:
    return DatabricksVolumesWriter(
        connector_config=SimpleDatabricksVolumesConfig(
            host= DATABRICKS_HOSTNAME ,                    
            access_config=DatabricksVolumesAccessConfig(
                username= DATABRICKS_USERNAME,              
                password= DATABRICKS_TOKEN                 
            ),
        ),
        write_config=DatabricksVolumesWriteConfig(
            catalog= catalog_name,                   
            schema = schema_name,                    
            volume= processed_output_volume_path     
        ),
    )



if __name__ == "__main__":
    writer = get_writer()
    runner = DropboxRunner(
        processor_config=ProcessorConfig(
            verbose=True,
            num_processes=2,
        ),
        read_config=ReadConfig(
            download_dir = download_dir_name
        ),
        partition_config=PartitionConfig(),
        connector_config=SimpleDropboxConfig(
            access_config=DropboxAccessConfig(
                token= DROPBOX_TOKEN                  
            ),   
            remote_url="dropbox:// /",
            recursive=True,
        ),
        writer=writer,
        writer_kwargs={},
       )
    runner.run()

# COMMAND ----------

#List the raw files on UC Volumes. On dropbox we had pdf's under a folder called databricks-pdf
dbutils.fs.ls(download_dir_name+"/databricks-pdf")


# COMMAND ----------


#List the json text files on UC Volumes. On dropbox we had pdf's under a folder called databricks-pdf
dbutils.fs.ls( processed_output_volume_full_path+"/databricks-pdf")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the data
# MAGIC select * from json.`/Volumes/demo_catalog/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io/structured-output/databricks-pdf/transform-scale-your-organization-with-data-ai-v16-052522.pdf.json`

# COMMAND ----------


download_dir_name = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/unstructured_io_api/raw"
processed_output_volume_path =f"{volume_name}/unstructured_io_api"
processed_output_volume_full_path = f"/Volumes/{catalog_name}/{schema_name}/{processed_output_volume_path}/structured-output"


# COMMAND ----------

# MAGIC %pip install unstructured-client==0.21.1

# COMMAND ----------

# DBTITLE 1,2. Using UnstructuredIO saas api to process files obtained using the source connector "Dropbox"
import os

from unstructured.ingest.connector.fsspec.dropbox import DropboxAccessConfig, SimpleDropboxConfig
from unstructured.ingest.interfaces import (
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import DropboxRunner


from unstructured.ingest.connector.databricks_volumes import (
    DatabricksVolumesAccessConfig,
    DatabricksVolumesWriteConfig,
    SimpleDatabricksVolumesConfig,
)
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import (
    ChunkingConfig,
    EmbeddingConfig,
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import LocalRunner
from unstructured.ingest.runner.writers.base_writer import Writer
from unstructured.ingest.runner.writers.databricks_volumes import (
    DatabricksVolumesWriter,
)


def get_writer() -> Writer:
    return DatabricksVolumesWriter(
        connector_config=SimpleDatabricksVolumesConfig(
            host= DATABRICKS_HOSTNAME ,                    
            access_config=DatabricksVolumesAccessConfig(
                username= DATABRICKS_USERNAME,              
                password= DATABRICKS_TOKEN                 
            ),
        ),
        write_config=DatabricksVolumesWriteConfig(
            catalog= catalog_name,                   
            schema = schema_name,                    
            volume= processed_output_volume_path     
        ),
    )

if __name__ == "__main__":
    writer = get_writer()
    runner = DropboxRunner(
        processor_config=ProcessorConfig(
            verbose=True,
            #output_dir="dropbox-output",
            num_processes=2,
        ),
        read_config=ReadConfig(
            download_dir = download_dir_name
        ),
        partition_config=PartitionConfig(
            partition_by_api=True,
            api_key=UNSTRUCTURED_API_KEY ,
            partition_endpoint= UNSTRUCTURED_SAAS_BASE_URL
            
        ),
        connector_config=SimpleDropboxConfig(
            access_config=DropboxAccessConfig(
                token= DROPBOX_TOKEN                 
            ), 
            remote_url="dropbox:// /",
            recursive=True,
        ),
        writer=writer,
        writer_kwargs={},
    )
    runner.run()

# COMMAND ----------

#List the raw files on UC Volumes. On dropbox we had pdf's under a folder called databricks-pdf
dbutils.fs.ls(download_dir_name+"/databricks-pdf")

# COMMAND ----------


#List the json text files on UC Volumes. On dropbox we had pdf's under a folder called databricks-pdf
dbutils.fs.ls( processed_output_volume_full_path+"/databricks-pdf")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the data
# MAGIC select * from json.`/Volumes/demo_catalog/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api/structured-output/databricks-pdf/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf.json`

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TEMPORARY VIEW unstructured_data_temp
# MAGIC USING json
# MAGIC OPTIONS (path="/Volumes/demo_catalog/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api/structured-output/databricks-pdf/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf.json",multiline=true)

# COMMAND ----------


download_dir_name = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/unstructured_io_api_3/raw"
processed_output_volume_path =f"{volume_name}/unstructured_io_api_3"
processed_output_volume_full_path = f"/Volumes/{catalog_name}/{schema_name}/{processed_output_volume_path}/structured-output"


# COMMAND ----------

# MAGIC %pip install "unstructured[embed-huggingface]"==0.12.6

# COMMAND ----------

# DBTITLE 1,3. Using UnstructuredIO oss to process a file on UC Volumes and automatically create embeddings

import os

from unstructured.ingest.connector.databricks_volumes import (
    DatabricksVolumesAccessConfig,
    DatabricksVolumesWriteConfig,
    SimpleDatabricksVolumesConfig,
)
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import (
    ChunkingConfig,
    EmbeddingConfig,
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import LocalRunner
from unstructured.ingest.runner.writers.base_writer import Writer
from unstructured.ingest.runner.writers.databricks_volumes import (
    DatabricksVolumesWriter,
)


def get_writer() -> Writer:
    return DatabricksVolumesWriter(
        connector_config=SimpleDatabricksVolumesConfig(
            host= DATABRICKS_HOSTNAME ,
            access_config=DatabricksVolumesAccessConfig(
                username= DATABRICKS_USERNAME,              
                password= DATABRICKS_TOKEN  
            ),
        ),
        write_config=DatabricksVolumesWriteConfig(
            catalog= catalog_name,                   
            schema = schema_name,                    
            volume= processed_output_volume_path     
        ),
    )


if __name__ == "__main__":
    writer = get_writer()
    runner = LocalRunner(
        processor_config=ProcessorConfig(
            verbose=True,
            output_dir="local-output-to-databricks-volumes",
            num_processes=2,
        ),
        connector_config=SimpleLocalConfig(
            input_path= "/Volumes/prasad_kona_dev/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api/raw/databricks-pdf/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf", 
        ),
        read_config=ReadConfig(),
        partition_config=PartitionConfig(),
        chunking_config=ChunkingConfig(chunk_elements=True),
        embedding_config=EmbeddingConfig(
            provider="langchain-huggingface",
        ),
        writer=writer,
        writer_kwargs={},
    )
    runner.run()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the data
# MAGIC select * from json.`/Volumes/demo_catalog/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api_3/local-output-to-databricks-volumes/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf.json`

# COMMAND ----------


download_dir_name = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/unstructured_io_api_4/raw"
processed_output_volume_path =f"{volume_name}/unstructured_io_api_4"
processed_output_volume_full_path = f"/Volumes/{catalog_name}/{schema_name}/{processed_output_volume_path}/structured-output"


# COMMAND ----------

# DBTITLE 1,4. Using UnstructuredIO Saas API to process a file on UC Volumes and automatically create embeddings

import os

from unstructured.ingest.connector.databricks_volumes import (
    DatabricksVolumesAccessConfig,
    DatabricksVolumesWriteConfig,
    SimpleDatabricksVolumesConfig,
)
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import (
    ChunkingConfig,
    EmbeddingConfig,
    PartitionConfig,
    ProcessorConfig,
    ReadConfig,
)
from unstructured.ingest.runner import LocalRunner
from unstructured.ingest.runner.writers.base_writer import Writer
from unstructured.ingest.runner.writers.databricks_volumes import (
    DatabricksVolumesWriter,
)


def get_writer() -> Writer:
    return DatabricksVolumesWriter(
        connector_config=SimpleDatabricksVolumesConfig(
            host= DATABRICKS_HOSTNAME ,
            access_config=DatabricksVolumesAccessConfig(
                username= DATABRICKS_USERNAME,              
                password= DATABRICKS_TOKEN  
            ),
        ),
        write_config=DatabricksVolumesWriteConfig(
            catalog= catalog_name,                   
            schema = schema_name,                    
            volume= processed_output_volume_path     
        ),
    )


if __name__ == "__main__":
    writer = get_writer()
    runner = LocalRunner(
        processor_config=ProcessorConfig(
            verbose=True,
            output_dir="local-output-to-databricks-volumes",
            num_processes=2,
        ),
        connector_config=SimpleLocalConfig(
            input_path= "/Volumes/prasad_kona_dev/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api/raw/databricks-pdf/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf", 
        ),
        read_config=ReadConfig(),
        partition_config=PartitionConfig(
            partition_by_api=True,
            api_key=UNSTRUCTURED_API_KEY ,
            partition_endpoint= UNSTRUCTURED_SAAS_BASE_URL
            
        ),        chunking_config=ChunkingConfig(chunk_elements=True),
        embedding_config=EmbeddingConfig(
            provider="langchain-huggingface",
        ),
        writer=writer,
        writer_kwargs={},
    )
    runner.run()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query the data
# MAGIC select * from json.`/Volumes/demo_catalog/unstructured_io_demo_schema/unstructured_io_demo_volume/unstructured_io_api_4/local-output-to-databricks-volumes/030521-2-The-Delta-Lake-Series-Complete-Collection.pdf.json`

# COMMAND ----------


