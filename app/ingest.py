"""
Ingestion pipeline: scrape docs -> chunk -> embed -> store in ChromaDB.

Usage:
    python ingest.py
"""

import config
from utils import CloudHubDocsScraper, DocumentChunker, ChromaDBManager


def main():
    # 1. Scrape
    print("=" * 60)
    print("STEP 1: SCRAPING DOCUMENTATION")
    print("=" * 60)
    scraper = CloudHubDocsScraper()
    scraped_docs = scraper.scrape()

    print(f"\n  Pages scraped : {len(scraped_docs)}")
    print(f"  Total chars   : {sum(d['length'] for d in scraped_docs):,}")

    if not scraped_docs:
        print("Nothing scraped — exiting.")
        return

    # 2. Chunk
    print("\n" + "=" * 60)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 60)
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(scraped_docs)

    # 3. Embed & store
    print("\n" + "=" * 60)
    print("STEP 3: STORING IN CHROMADB")
    print("=" * 60)
    db = ChromaDBManager()
    db.add_chunks(chunks)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Collection : {config.COLLECTION_NAME}")
    print(f"  DB path    : {config.VECTOR_DB_PATH}")
    print(f"  Documents  : {db.collection.count()}")


if __name__ == "__main__":
    main()
