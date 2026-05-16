# LangChain Overview

LangChain is a framework for developing applications powered by large language models (LLMs).

## Core Concepts

### LLMs and Chat Models
LangChain provides a standard interface for interacting with LLMs. The two primary types are:
- **LLMs**: Take a string as input and return a string
- **Chat Models**: Take a list of messages as input and return a message

### Chains
Chains are sequences of calls to LLMs or other utilities. A chain takes input, passes it through a series of steps, and returns output. The simplest chain is an LLMChain, which combines a prompt template with an LLM.

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01"
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"topic": "retrieval augmented generation"})
```

### Prompt Templates
Prompt templates are reusable, parameterized prompts. They allow you to define a template with variables and fill them in at runtime.

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about {domain}."),
    ("human", "{question}")
])

formatted = template.format_messages(domain="Python", question="What is a decorator?")
```

### Memory
LangChain provides memory modules to persist state across LLM calls. This is critical for building conversational applications.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "Hello!"}, {"output": "Hi there, how can I help?"})
history = memory.load_memory_variables({})
```

### Document Loaders
Document loaders read data from different sources and return a list of Document objects. Common loaders include:
- `TextLoader` for plain text files
- `PyPDFLoader` for PDF documents
- `WebBaseLoader` for web pages
- `UnstructuredMarkdownLoader` for Markdown files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("my_document.txt")
documents = loader.load()
```

### Text Splitters
Text splitters divide long documents into smaller chunks suitable for embedding and retrieval.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

## Installation

```bash
pip install langchain langchain-openai langchain-community
```

## Key Principles
1. **Composability**: Build complex behaviour from simple, reusable components
2. **Flexibility**: Swap LLM providers, vector stores, and memory backends without rewriting application logic
3. **Observability**: Built-in tracing and logging via LangSmith
