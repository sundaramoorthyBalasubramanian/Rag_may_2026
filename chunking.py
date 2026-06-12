import re

def split_into_blocks(text):
    """
    Splits content into:
    - table blocks
    - image blocks
    - normal text
    """
    # Regex for markdown tables
    table_pattern = r'(\|.*\|(?:\n\|.*\|)+)'
    
    # Regex for your image placeholder
    image_pattern = r'(\[Image page:.*?\])'

    # Combine patterns
    pattern = f'{table_pattern}|{image_pattern}'

    parts = re.split(pattern, text)

    return [p.strip() for p in parts if p and p.strip()]

def get_block_type(block):
    if block.startswith("[Image page"):
        return "image"
    elif "|" in block and "\n" in block:
        return "table"
    else:
        return "text"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_document(processed_pages, debug=False):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=30
    )

    all_chunks = []

    for page in processed_pages:
        blocks = split_into_blocks(page["content"])

        for block in blocks:
            block_type = get_block_type(block)

            # -------------------------
            # TABLE → keep intact
            # -------------------------
            if block_type == "table":
                chunk = Document(
                    page_content="Table:\n" + block,
                    metadata={
                        "page": page["page"],
                        "content_type": page["type_summary"],
                        "block_type": "table"
                    }
                )
                all_chunks.append(chunk)

            # -------------------------
            # IMAGE → keep as single chunk
            # -------------------------
            elif block_type == "image":
                chunk = Document(
                    page_content=block,
                    metadata={
                        "page": page["page"],
                        "content_type": page["type_summary"],
                        "block_type": "image"
                    }
                )
                all_chunks.append(chunk)

            # -------------------------
            # TEXT → split normally
            # -------------------------
            else:
                splits = text_splitter.split_text(block)

                for s in splits:
                    if len(s.strip()) < 80:
                        continue

                    chunk = Document(
                        page_content=s,
                        metadata={
                            "page": page["page"],
                            "content_type": page["type_summary"],
                            "block_type": "text"
                        }
                    )
                    all_chunks.append(chunk)

                    if debug:
                        print("\n--- TEXT CHUNK ---")
                        print(s[:200])

    return all_chunks           

def check_table_chunks(chunks):
    for c in chunks:
        if c.metadata["block_type"] == "table":
            print("\n--- TABLE CHUNK ---")
            print(c.page_content)    