## Process semi-structured data using Large Language Models (LLMs) on Amazon Bedrock

This repository contains samples to process semi-structured data (XML, CSV and JSON) using Large Language Models (LLMs) on Amazon Bedrock.

### Overview

[Semi-structured data](https://en.wikipedia.org/wiki/Semi-structured_data) is a form of structured data that does not obey the tabular structure of data models associated with relational databases or other forms of data tables, but nonetheless contains tags or other markers to separate semantic elements and enforce hierarchies of records and fields within the data.

In complex [Generative AI](https://en.wikipedia.org/wiki/Generative_artificial_intelligence) use cases that involve [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model), we often come across the need to process semi-structured data through natural language queries. These data operations would involve,

* Data extraction with conditions
* Filtering
* Aggregation
* Sorting
* Transformations
* Sample data generation from XML schemas
* Sample API request generation from a Swagger JSON document

The samples in this repository will demonstrate how to do these using the LLMs on [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) which is a fully managed service that makes high-performing foundation models (FMs) from leading AI startups and Amazon available for your use through a unified API. After running through these, you will learn what data operations can be effectively performed using LLMs and what operations may not be a good fit for LLMs.

Note:

* These notebooks should only be run from within an [Amazon SageMaker Notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) or within an [Amazon SageMaker Studio Notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html)
* These notebooks use text based models along with their versions that were available on Amazon Bedrock at the time of writing. Update these as required.
* At the time of this writing, Amazon Bedrock was only available in [these supported AWS Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html). If you are running these notebooks from any other AWS Region, then you have to change the Amazon Bedrock client's region and/or endpoint URL parameters to one of those supported AWS Regions. Follow the guidance in the *Organize imports* section of the individual notebooks.
* On an Amazon SageMaker Notebook instance, these notebooks are recommended to be run with a minimum instance size of *ml.m5.xlarge* and with *Amazon Linux 2, Jupyter Lab 3* as the platform identifier.
* On an Amazon SageMaker Studio Notebook, these notebooks are recommended to be run with a minimum instance size of *ml.m5.xlarge* and with *Data Science 3.0* as the image.
* At the time of this writing, the most relevant latest version of the Kernel for running these notebooks were *conda_python3* on an Amazon SageMaker Notebook instance or *Python 3* on an Amazon SageMaker Studio Notebook.

### Repository structure

This repository contains

* [A Jupyter Notebook for XML processing](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/llm_xml_data_processing.ipynb).
* [A Jupyter Notebook for CSV processing](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/llm_csv_data_processing.ipynb).
* [A Jupyter Notebook for JSON processing](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/llm_json_data_processing.ipynb).

* [A Python script](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/scripts/helper_functions.py) that contains helper functions.

* [A prompt templates folder](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/prompt_templates) for XML, CSV and JSON data formats across various LLMs.

* [A data folder](https://github.com/aws-samples/semi-structured-data-processing-with-amazon-bedrock/blob/main/notebooks/data) for the data used in various prompts across  XML, CSV and JSON data formats and across various LLMs.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
