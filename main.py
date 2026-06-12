import argparse
import os
import dataIngestion 
from chunking import chunk_document,chunk_document,check_table_chunks
from vector_db import save_to_db, load_db,rag_query
from dotenv import load_dotenv

def main():
    load_dotenv()
    debug =True
    parser = argparse.ArgumentParser(
        description="Process .pdf or .mp4 files into structured text chunks for RAG."
    )
    
    # Positional argument for the file target
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the local .pdf or .mp4 file."
    )
    
    # Optional parameters to adjust chunk properties
    parser.add_argument("--size", type=int, default=500, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks.")

    args = parser.parse_args()

    # Verify target file exists
    if not os.path.exists(args.file_path):
        print(f"Error: The path '{args.file_path}' does not point to a valid file.")
        return

    # Check the file extension
    ext = os.path.splitext(args.file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            print(f"Processing PDF document: {args.file_path}...")
            results = dataIngestion.process_pdf(args.file_path)
            if debug:
                print(f"@Total pages processed: {len(results)}")
                #evaluate_ingestion(results)
                #sample_pages(results)

            chunks = chunk_document(results)
            if debug:
                print(f"Total chunks created: {len(chunks)}")
                #
                check_table_chunks(chunks)
            save_to_db(chunks)
            loaded_db = load_db()
            print("\nload_db done ===")
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset

            query_string = "Alan Trevor's look like?"
            data = {
                "question": [query_string],
                "answer": [rag_query(loaded_db, query_string)],
                "contexts": [[doc.page_content for doc in loaded_db.similarity_search(query_string, k=3)]],
                "ground_truth": ["rough fellow and red beard face..."]
            }

            dataset = Dataset.from_dict(data)
            print("\ndataset done ===")

            from langchain_openai import OpenAIEmbeddings
            from ragas import evaluate

            embeddings = OpenAIEmbeddings()
            print("\embeddings done ===")
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                embeddings=embeddings
            )
            print("\n=== RAG Evaluation Results ===")
            print(result)
            query = "any moral from story?"
            answer = rag_query(loaded_db, query)
            print("Final answer:", answer)
    
        elif ext == ".mp4":
            print(f"Processing MP4 media file: {args.file_path} (Converting to text via Whisper)...")
            
            
        else:
            print(f"Error: Format '{ext}' is unsupported. Please supply a .pdf or .mp4 file.")
            return

        # Output basic metrics
        print("\n=== Processing Complete ===")
       
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()
