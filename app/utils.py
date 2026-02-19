from openai import OpenAI
import os
import re
import time
import chromadb
import tiktoken
import requests
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class CloudHubDocsScraper:
    """Recursively crawls a documentation site and extracts clean text."""

    def __init__(
        self,
        base_url: str = config.SCRAPE_BASE_URL,
        max_pages: int = config.SCRAPE_MAX_PAGES,
        delay: float = config.SCRAPE_DELAY,
    ):
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []

        parsed = urlparse(base_url)
        self.allowed_domain = parsed.netloc
        self.allowed_path_prefix = parsed.path

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != self.allowed_domain:
            return False
        if not parsed.path.startswith(self.allowed_path_prefix):
            return False
        skip_extensions = [".pdf", ".zip", ".png", ".jpg",
                           ".jpeg", ".gif", ".css", ".js", ".xlsx"]
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        return True

    def fetch_page(self, url: str) -> str | None:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Educational RAG Project)"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None

    def extract_text_and_links(self, html: str, url: str) -> tuple:
        soup = BeautifulSoup(html, "lxml")
        BeautifulSoup()
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "No Title"

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=re.compile("content|main|article", re.I))
            or soup.find("body")
        )

        text = ""
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
            text = re.sub(r"\n\s*\n", "\n\n", text)

        links: Set[str] = set()
        for link in soup.find_all("a", href=True):
            absolute_url = urljoin(url, link["href"]).split("#")[
                0].split("?")[0]
            if self.is_valid_url(absolute_url):
                links.add(absolute_url)

        return title_text, text, links

    def scrape(self, start_url: str | None = None) -> List[Dict]:
        if start_url is None:
            start_url = self.base_url

        urls_to_visit = [start_url]

        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in self.visited_urls:
                continue

            print(
                f"Scraping [{len(self.visited_urls) + 1}/{self.max_pages}]: {current_url}")

            html = self.fetch_page(current_url)
            if not html:
                self.visited_urls.add(current_url)
                continue

            title, text, links = self.extract_text_and_links(html, current_url)

            if text and len(text.strip()) > 100:
                self.scraped_data.append(
                    {"url": current_url, "title": title,
                        "text": text, "length": len(text)}
                )

            self.visited_urls.add(current_url)
            for link in links:
                if link not in self.visited_urls:
                    urls_to_visit.append(link)

            time.sleep(self.delay)

        print(
            f"\nScraping complete! Collected {len(self.scraped_data)} pages.")
        return self.scraped_data


class DocumentChunker:
    """Splits documents into smaller pieces for embedding."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, scraped_data: List[Dict]) -> List[Dict]:
        all_chunks: List[Dict] = []
        for doc in scraped_data:
            text_chunks = self.text_splitter.split_text(doc["text"])
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append(
                    {
                        "text": chunk_text,
                        "source_url": doc["url"],
                        "source_title": doc["title"],
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    }
                )
        print(
            f"Created {len(all_chunks)} chunks from {len(scraped_data)} documents")
        return all_chunks


class ChromaDBManager:
    """Manages ChromaDB operations including embeddings and storage."""

    def __init__(
        self,
        persist_directory: str = config.VECTOR_DB_PATH,
        collection_name: str = config.COLLECTION_NAME,
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "MuleSoft CloudHub documentation for RAG"},
        )

        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or environment.")
        self.openai_client = OpenAI(api_key=api_key)

    def generate_embedding(
        self, text: str, model: str = config.EMBEDDING_MODEL
    ) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text, model=model)
        return response.data[0].embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        max_tokens: int = config.EMBEDDING_MAX_TOKENS,
    ) -> List[List[float] | None]:
        encoding = tiktoken.encoding_for_model(model)
        all_embeddings: List[List[float] | None] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            processed_batch = []

            for text in batch:
                tokens = encoding.encode(text)
                if len(tokens) > max_tokens:
                    print(
                        f"  Warning: text truncated from {len(tokens)} to {max_tokens} tokens")
                    text = encoding.decode(tokens[:max_tokens])
                processed_batch.append(text)

            print(
                f"  Generating embeddings — batch {i // batch_size + 1}"
                f"/{(len(texts) - 1) // batch_size + 1}"
            )

            try:
                response = self.openai_client.embeddings.create(
                    input=processed_batch, model=model)
                all_embeddings.extend(
                    [item.embedding for item in response.data])
                time.sleep(0.5)
            except Exception as e:
                print(f"  Error in batch {i // batch_size + 1}: {e}")
                all_embeddings.extend([None] * len(batch))

        return all_embeddings

    def add_chunks(
        self, chunks: List[Dict], batch_size: int = config.EMBEDDING_BATCH_SIZE
    ) -> None:
        print(f"\nAdding {len(chunks)} chunks to ChromaDB ...")
        texts = [c["text"] for c in chunks]

        print("Generating embeddings ...")
        embeddings = self.generate_embeddings_batch(
            texts, batch_size=batch_size)

        valid = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
        print(f"Successfully generated {len(valid)} embeddings")

        for i in range(0, len(valid), batch_size):
            batch = valid[i: i + batch_size]
            self.collection.add(
                ids=[f"chunk_{j}" for j in range(i, i + len(batch))],
                documents=[item[0]["text"] for item in batch],
                embeddings=[item[1] for item in batch],
                metadatas=[
                    {"source_url": item[0]["source_url"],
                        "source_title": item[0]["source_title"]}
                    for item in batch
                ],
            )
            print(
                f"  Stored batch {i // batch_size + 1}/{(len(valid) - 1) // batch_size + 1}")

        print(f"Total documents in collection: {self.collection.count()}")

    def query(self, query_text: str, n_results: int = config.MAX_RETRIEVAL_CHUNKS):
        query_embedding = self.generate_embedding(query_text)
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)


class RAGAssistant:
    """Retrieval-Augmented Generation over the vector store."""

    def __init__(
        self,
        db_manager: ChromaDBManager,
        max_context_tokens: int = config.MAX_CONTEXT_TOKENS,
    ):
        self.db_manager = db_manager
        self.max_context_tokens = max_context_tokens

    def retrieve_context(self, query: str, max_chunks: int = config.MAX_RETRIEVAL_CHUNKS):
        results = self.db_manager.query(query, n_results=max_chunks)
        context_parts: List[str] = []
        sources: List[Dict] = []
        total_tokens = 0

        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk_tokens = len(doc) // 4
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            context_parts.append(
                f"{doc}\n[Source: {metadata['source_title']}]")
            sources.append(metadata)
            total_tokens += chunk_tokens

        return {
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
            "tokens_used": total_tokens,
            "chunks_used": len(context_parts),
        }

    def ask(
        self,
        question: str,
        chat_history: list | None = None,
        model: str = config.LLM_MODEL,
    ):
        retrieval = self.retrieve_context(question)

        system_prompt = (
            "You are a knowledgeable assistant specialized in MuleSoft CloudHub documentation.\n"
            "Your responsibilities:\n"
            "1. Answer questions using ONLY the provided documentation context\n"
            "2. Cite sources using the [Source: ...] format provided in the context\n"
            "3. If the context doesn't contain enough information to answer, clearly state that\n"
            "4. Be concise but thorough in your explanations\n"
            "5. Include relevant URLs from the sources when helpful\n"
            "6. Maintain a professional, helpful tone"
        )

        user_prompt = (
            f"DOCUMENTATION CONTEXT:\n{retrieval['context']}\n\n---\n\n"
            f"USER QUESTION:\n{question}\n\n---\n\n"
            "Provide a clear, accurate answer based on the documentation context above. "
            "Always cite your sources."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})

        response = self.db_manager.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": retrieval["sources"],
            "chunks_used": retrieval["chunks_used"],
            "tokens_used": retrieval["tokens_used"],
        }
