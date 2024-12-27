# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import hashlib
from typing import List, Union

import chromadb
from pydantic import Field

from mbodied.agents import Agent
from mbodied.types.sample import Sample


class Document(Sample):
    """Sample for documents to be upserted into the collection."""

    id: str | None = Field(
        default=None,
        description="Unique identifier for the document (auto-generated if not provided)",
    )
    content: str = Field(..., description="The textual content of the document")
    metadata: dict | None = Field(default=None, description="Optional metadata to associate with the document")

    def __init__(self, **data):
        """Override the default constructor to generate a hash-based ID if `id` is not provided.

        Same content will result in same hash. This can avoid duplication.
        """
        super().__init__(**data)
        if self.id is None:
            # Generate a hash of the content as the ID
            self.id = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class RagAgent(Agent):
    r"""A Retrieval-Augmented Generation (RAG) agent using Chroma as the vector store.

    The agent can upsert documents from disk, query for relevant documents, and construct a prompt.

    Example:
        ```python
        # 1. Instantiate RagAgent
        agent = RagAgent(collection_name="my_collection", path="./chroma", distance_threshold=1.5)

        # 2. Upsert various types of inputs
        # Single string
        agent.upsert("This is a single document as a string.")

        # List of strings
        agent.upsert(["Document 1", "Document 2", "Document 3"])

        # List of Document objects
        documents = [
            Document(content="This is a document about remote controls."),
            Document(content="This is another document about boxes."),
            Document(id="id_example", content="This is another document about shoes."),
            Document(
                content="This document has metadata.",
                metadata={"type": "example"},
            ),
        ]

        agent.upsert(documents)

        # 3. Use act to query and generate a response
        user_query_text = "Tell me about remote controls."
        response_prompt = agent.act(user_query=user_query_text, n_results=2)
        print("Constructed prompt:\n", response_prompt)
        ```
    """

    def __init__(
        self,
        collection_name: str = "my_collection",
        path: str = "./chroma",
        distance_threshold: float = 1.5,
        **chroma_config,
    ):
        """Initialize the RagAgent with the collection name, path, and distance threshold.

        Args:
            collection_name: Name of the collection.
            path: Path to the Chroma database on disk.
            distance_threshold: Distance threshold for filtering out similar documents.
            **chroma_config: Additional configuration options for the Chroma client.
        """
        self.collection_name = collection_name
        self.path = path
        self.distance_threshold = distance_threshold

        # Initialize the Chroma persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.path, **chroma_config)

        # Get or create the collection
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def upsert(
        self,
        documents: List[Document] | List[str] | str,
        dedup: bool = False,
        dedup_threshold: float | None = 0.1,
    ) -> None:
        """Upsert a list of documents or strings, or a single string into the collection.

        if dedup is enabled, similar documents will be filtered out based on the dedup_threshold.

        Args:
            documents: A list of `Document` objects, a list of strings, or a single string.
            dedup: Enable deduplication of similar documents.
            dedup_threshold: Distance threshold for deduplication.
        """
        # Normalize the input into a list of `Document` objects
        if isinstance(documents, str):
            documents = [Document(content=documents)]
        elif isinstance(documents, list):
            if all(isinstance(doc, str) for doc in documents):
                documents = [Document(content=doc) for doc in documents]
            elif not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("Documents must be a list of `Document` objects or strings.")

        # Filter documents if dedup is enabled
        if dedup:
            filtered_documents = []
            for doc in documents:
                # Query the collection to check for similar embeddings
                results = self.collection.query(query_texts=[doc.content], n_results=1)
                if len(results["documents"]) == 0 or results["distances"][0][0] > dedup_threshold:
                    # Add only if no similar document exists
                    filtered_documents.append(doc)
            documents = filtered_documents

        # Extract data for upsert
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents] if any(doc.metadata for doc in documents) else None

        # Perform the upsert operation
        if documents:
            self.collection.upsert(documents=contents, ids=ids, metadatas=metadatas)

    async def async_upsert(self, documents: List[Document] | List[str] | str) -> None:
        """Upsert documents asynchronously using asyncio."""
        await asyncio.to_thread(self.upsert, documents=documents)

    def query(
        self,
        query_texts: List[str],
        n_results: int = 2,
        distance_threshold: float | None = None,
    ) -> List[List[str]]:
        """Query the embedded documents in the collection and optionally filter by distance.

        Args:
            query_texts: List of query strings.
            n_results: Number of documents to retrieve.
            distance_threshold: Filter out results above this distance score.
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        results = self.collection.query(query_texts=query_texts, n_results=n_results)
        # Filter out any documents whose distance is above distance_threshold
        filtered_docs_per_query = []
        for docs_per_query, dists_per_query in zip(results["documents"], results["distances"]):
            filtered_docs = [doc for doc, dist in zip(docs_per_query, dists_per_query) if dist < distance_threshold]
            filtered_docs_per_query.append(filtered_docs)

        return filtered_docs_per_query

    def act(
        self,
        user_query: str,
        n_results: int = 2,
        distance_threshold: float | None = None,
        custom_prompt: str | None = None,
    ) -> str:
        """Handles both querying the collection and constructing the final prompt.

        Args:
            user_query: The user's query text.
            n_results: Number of context documents to retrieve.
            distance_threshold: Filter out results above this distance score.
            custom_prompt: Optional custom prompt to prepend to the response.
        """
        # Query for relevant documents
        relevant_docs = self.query(
            query_texts=[user_query],
            n_results=n_results,
            distance_threshold=distance_threshold,
        )
        context_docs = relevant_docs[0] if relevant_docs else []

        if len(context_docs) == 0:
            return user_query

        prompt = ""
        if custom_prompt:
            prompt += f"{custom_prompt}\n"
        prompt += (
            "Below are some context documents:\n\n"
            f"{'-'*40}\n"
            f"{chr(10).join(context_docs)}\n"
            f"{'-'*40}\n"
            "Here's the User's query:\n\n"
            f"{user_query}"
        )

        return prompt

    def delete_all(self) -> None:
        """Clean up the whole collection.

        Be cautious when using this method. It will delete all documents in the collection on disk.
        """
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(name=self.collection_name)


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # 1. Instantiate RagAgent
    agent = RagAgent(collection_name="my_collection", path="./chroma", distance_threshold=1.5)

    # 2. Upsert various types of inputs
    # Single string
    agent.upsert("This is a single document as a string.")

    # List of strings
    agent.upsert(["Document 1", "Document 2", "Document 3"])

    # List of Document objects
    documents = [
        Document(content="This is a document about remote controls."),
        Document(content="This is another document about boxes."),
        Document(id="id_example", content="This is another document about shoes."),
        Document(
            content="This document has metadata.",
            metadata={"type": "example"},
        ),
    ]
    agent.upsert(documents)

    # 3. Use act to query and generate a response
    user_query_text = "Tell me about remote controls."
    response_prompt = agent.act(user_query=user_query_text, n_results=2)
    print("Constructed prompt:\n", response_prompt)
