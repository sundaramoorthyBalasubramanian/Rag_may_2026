import argparse
import os
import uploadFile

def main():
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
            chunks = uploadFile.process_pdf(args.file_path, args.size, args.overlap)
            
        elif ext == ".mp4":
            print(f"Processing MP4 media file: {args.file_path} (Converting to text via Whisper)...")
            chunks = uploadFile.process_mp4(args.file_path, args.size, args.overlap)
            
        else:
            print(f"Error: Format '{ext}' is unsupported. Please supply a .pdf or .mp4 file.")
            return

        # Output basic metrics
        print("\n=== Processing Complete ===")
        print(f"Successfully generated {len(chunks)} text chunks.")
        if chunks:
            print(f"Sample Chunk: \"{chunks[0][:120]}...\"")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()
