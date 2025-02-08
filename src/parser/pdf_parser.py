from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import os
import re
import time
import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from src.parser.base_parser import BaseParser
from logger import logger

class PDFParser(BaseParser):
    """
    Parser for PDF files using docling
    """
    def __init__(self, artifacts_path, ocr_model_path = None) -> None:
        self.ocr_model_path = ocr_model_path
        self.artifacts_path = artifacts_path

        if ocr_model_path != None:
            self.use_ocr = True
        else:
            self.use_ocr = False
        self._init_pdf_parser(self.use_ocr)

    def _init_pdf_parser(self, use_ocr = True):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.artifacts_path = self.artifacts_path
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.do_ocr = use_ocr
        
        if use_ocr == True:
            easy_ocr_option = EasyOcrOptions()
            easy_ocr_option.model_storage_directory = self.ocr_model_path
            easy_ocr_option.lang = ['en', 'ch_sim']
            pipeline_options.ocr_options  = easy_ocr_option
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, file_path_or_url, output_dir, ouput_type = 'md'):
        start_time = time.time()
        result = self.doc_converter.convert(file_path_or_url)
        end_time = time.time() - start_time
        logger.info(f"Document converted in {end_time:.2f} seconds.")

        ## Export results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = result.input.file.stem

        try:
            if ouput_type == 'json':
                with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
                    fp.write(json.dumps(result.document.export_to_dict()))
            elif ouput_type == 'text':
                with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
                    fp.write(result.document.export_to_text())
            elif ouput_type == 'md':
                with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
                    fp.write(result.document.export_to_markdown())
            elif ouput_type == 'doctags':
                with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
                    fp.write(result.document.export_to_document_tokens())
        except:
            print(f"Failed to export {doc_filename} to {ouput_type}.")

        return result