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


import pytest
import os
from pathlib import Path

from mbodied.agents.language.rag_agent import Document, RagAgent


@pytest.fixture
def agent(tmp_path):
    """
    Fixture to initialize RagAgent with a temporary Chroma path.
    This ensures each test uses an isolated, fresh Chroma DB.
    """
    collection_name = "test_collection"
    chroma_path = str(tmp_path / "chroma_db")
    # Create the agent
    _agent = RagAgent(collection_name=collection_name, path=chroma_path, distance_threshold=1.5)
    yield _agent
    # Cleanup after test
    if os.path.exists(chroma_path):
        # Danger: This deletes the entire directory where the test data is stored
        for item in Path(chroma_path).iterdir():
            if item.is_dir():
                for sub_item in item.iterdir():
                    sub_item.unlink()
                item.rmdir()
            else:
                item.unlink()
        Path(chroma_path).rmdir()


def test_document_id_generation():
    """
    Test that Document automatically generates the same ID for the same content
    and different IDs for different content.
    """
    doc1 = Document(content="Hello world")
    doc2 = Document(content="Hello world")
    doc3 = Document(content="Something else")

    # Same content -> same ID
    assert doc1.id == doc2.id, "Documents with identical content should have identical IDs"

    # Different content -> different ID
    assert doc1.id != doc3.id, "Documents with different content should have different IDs"


def test_upsert_single_string(agent):
    """
    Test upserting a single string document.
    """
    text = "This is a single test document."
    agent.upsert(text)

    # Query
    results = agent.query(["test document"], n_results=1)
    assert len(results) == 1, "Expected one list of results"
    assert len(results[0]) == 1, "Expected exactly one matching document"
    assert text in results[0][0], "The returned document should contain the original text"


def test_upsert_multiple_strings(agent):
    """
    Test upserting a list of string documents.
    """
    texts = ["Doc 1 about apples", "Doc 2 about oranges", "Doc 3 about bananas"]
    agent.upsert(texts)

    # Query
    results = agent.query(["apples"], n_results=1)
    assert len(results) == 1
    assert len(results[0]) == 1, "Expected one match for 'apples'"
    assert "apples" in results[0][0], "The returned doc should be about apples"


def test_upsert_document_objects(agent):
    """
    Test upserting a list of Document objects.
    """
    documents = [
        Document(content="First document about cars"),
        Document(content="Second document about bikes"),
    ]
    agent.upsert(documents)

    # Query
    results = agent.query(["cars"], n_results=1)
    assert len(results) == 1
    assert len(results[0]) == 1
    assert "cars" in results[0][0], "Should return the document about cars"


def test_upsert_dedup(agent):
    """
    Test the deduplication feature.
    We upsert the same document twice with dedup=True and expect only one entry.
    """
    doc_text = "Document about duplication."
    agent.upsert(doc_text)

    # Upsert the same document again with dedup=True
    agent.upsert(doc_text, dedup=True, dedup_threshold=0.9)

    # Query
    results = agent.query(["duplication"], n_results=2)
    # Should return just one document
    assert len(results[0]) == 1, "Deduplication should prevent inserting the same document twice"


def test_act_no_context(agent):
    """
    If no documents match, act() should return just the user_query.
    """
    user_query = "Non-existent context"
    prompt = agent.act(user_query=user_query, n_results=1)
    assert prompt == user_query, "If no relevant docs, act() should just return the user_query"


def test_act_with_context(agent):
    """
    If documents are found, they should be included in the constructed prompt.
    """
    doc_texts = ["Remote controls are very useful", "Boxes are square"]
    agent.upsert(doc_texts)

    user_query = "Tell me about remote controls."
    prompt = agent.act(user_query=user_query, n_results=2)

    assert "Remote controls are very useful" in prompt, "Prompt should contain the matching doc about remote controls"
    assert user_query in prompt, "Prompt should contain the user's query"


def test_delete_all(agent):
    """
    Test that delete_all() clears the collection.
    """
    # Upsert some docs
    agent.upsert(["Temporary doc 1", "Temporary doc 2"])
    results_before = agent.query(["doc"], n_results=2)
    assert len(results_before[0]) == 2, "Should have 2 documents initially"

    # Now delete everything
    agent.delete()
