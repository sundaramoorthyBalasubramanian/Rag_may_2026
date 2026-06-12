import fitz  # PyMuPDF
from pdf2image import convert_from_path
import os
import easyocr
import numpy as np

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

def sample_pages(results, n=3):
    for i in range(min(n, len(results))):
        page = results[i]
        print("\n--- SAMPLE PAGE ---")
        print("Page:", page["page"])
        print("Type:", page["type_summary"])
        print(page["content"][:500])

def evaluate_ingestion(results):
    total_pages = len(results)
    empty_pages = 0
    noisy_pages = 0
    ocr_pages = 0

    for page in results:
        text = page["content"]

        # Empty check
        if len(text.strip()) == 0:
            empty_pages += 1

        # Quality check
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if page["type_summary"] == "text + table":
            print("Page:", page["page"])
            print("alpha_ratio:", round(alpha_ratio, 2))
            print("Type:", page["type_summary"])
            print(page["content"])


        if alpha_ratio < 0.2:
            print(f"@@@@@@@Page {page['page']} → alpha_ratio = {round(alpha_ratio, 2)}")
            noisy_pages += 1
                    
        # OCR tracking
        if "scanned_image_ocr" in page["type_summary"]:
            ocr_pages += 1

    print("\n=====INGESTION REPORT =====")
    print("Total pages:", total_pages)
    print("Empty pages:", empty_pages)
    print("Noisy pages:", noisy_pages)
    print("OCR pages:", ocr_pages)

    print("\nQuality %:", round((total_pages - noisy_pages) / total_pages * 100, 2))

def process_pdf(pdf_path):
    # 1. Initialize EasyOCR for the pure image fallback
    reader = easyocr.Reader(['en'])

    # 2. Configure Docling with EasyOCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.ocr_options = EasyOcrOptions()
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page_info = {"page": page_num + 1, "types": [], "content": ""}
        print(f"Processing Page {page_num + 1}...")

        # Step 1: Convert page with Docling (Handles Text and Tables)
        conv_result = converter.convert(pdf_path, page_range=(page_num + 1, page_num + 1))
        doc_obj = conv_result.document
        
        # Step 2: Detect content types using Docling's internal model
        has_text = any(item for item, _ in doc_obj.iterate_items() if hasattr(item, 'text') and item.text.strip())
        has_table = len(doc_obj.tables) > 0
        
        # Step 3: PyMuPDF Check for actual embedded images
        has_image = len(doc[page_num].get_images()) > 0

        # Step 4: Logic Gate for Labeling
        if has_text: page_info["types"].append("text")
        if has_table: page_info["types"].append("table")
        if has_image: page_info["types"].append("image")

        # Step 5: Content Extraction based on detected mixture
        if has_text or has_table:
            # Docling handles the "mixture" of text and tables perfectly in Markdown
            page_info["content"] = doc_obj.export_to_markdown()
        
        # Fallback: If it's a scanned page (No text/tables found by Docling)
        if not has_text and not has_table:
            images = convert_from_path(pdf_path, dpi=300, first_page=page_num+1, last_page=page_num+1)
            ocr_text = ""
            for img in images:
                img_np = np.array(img)
                ocr_results = reader.readtext(img_np, detail=0)
                ocr_text += " ".join(ocr_results)
            page_info["content"] = ocr_text
            page_info["types"] = ["scanned_image_ocr"]

        # Final Type Labeling
        page_info["type_summary"] = " + ".join(page_info["types"]) if page_info["types"] else "empty"
        results.append(page_info)
        
        print(f"Page {page_num + 1}: [{page_info['type_summary']}]")

    return results


