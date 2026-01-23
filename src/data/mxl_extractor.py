"""
MXL File Extractor

MusicXML (.mxl) files are compressed ZIP archives containing XML files.
This module handles decompression and extraction of the actual MusicXML content.
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List
from lxml import etree
from tqdm import tqdm


class MXLExtractor:
    """Extract and process .mxl (compressed MusicXML) files."""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the MXL extractor.

        Args:
            input_dir: Directory containing .mxl files
            output_dir: Directory to write extracted .xml files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_mxl_files(self) -> List[Path]:
        """Find all .mxl files in the input directory."""
        return list(self.input_dir.glob("**/*.mxl"))

    def extract_single(self, mxl_path: Path) -> Optional[str]:
        """
        Extract a single .mxl file and return the MusicXML content.

        Args:
            mxl_path: Path to .mxl file

        Returns:
            MusicXML content as string, or None if extraction fails
        """
        try:
            # Create temp directory for extraction
            temp_dir = self.output_dir / "temp" / mxl_path.stem
            temp_dir.mkdir(parents=True, exist_ok=True)

            # .mxl files are ZIP archives
            with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the XML file (usually in root of extracted content)
            xml_content = None
            for xml_file in temp_dir.rglob("*.xml"):
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if it's actual MusicXML (has score-partwise or timewise)
                    if 'score-partwise' in content or 'score-timewise' in content:
                        xml_content = content
                        break

            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

            return xml_content

        except Exception as e:
            print(f"Error extracting {mxl_path}: {e}")
            return None

    def extract_all(self) -> int:
        """
        Extract all .mxl files in the input directory.

        Returns:
            Number of successfully extracted files
        """
        mxl_files = self.find_mxl_files()
        print(f"Found {len(mxl_files)} .mxl files")

        success_count = 0
        for mxl_path in tqdm(mxl_files, desc="Extracting MXL files"):
            xml_content = self.extract_single(mxl_path)

            if xml_content:
                # Write to output with same basename
                output_path = self.output_dir / f"{mxl_path.stem}.xml"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                success_count += 1

        print(f"Successfully extracted {success_count}/{len(mxl_files)} files")
        return success_count


def normalize_musicxml(xml_content: str) -> str:
    """
    Normalize MusicXML by removing layout-specific elements.

    Keeps only the musical content (notes, pitches, rhythms, etc.)
    Removes: formatting, page layout, positioning, credits

    Args:
        xml_content: Raw MusicXML string

    Returns:
        Normalized MusicXML string
    """
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))

        # Define tags to remove (layout/formatting)
        tags_to_remove = [
            '{http://www.musicxml.org/ns/lilypond}*-body',
            # Page layout elements
            './/page-layout',
            './/system-layout',
            './/system-margins',
            './/top-system-distance',
            './/system-distance',
            # Scaling
            './/scaling',
            './/millimeters',
            './/tenths',
            # Appearance
            './/appearance',
            './/line-width',
            './/note-size',
            './/distance',
            # Print elements
            './/print',
            './/page-height',
            './/page-width',
            './/page-margins',
            # Credits
            './/credit',
            './/credit-words',
            # Formatting defaults we don't need
            './/word-font',
            './/lyric-font',
        ]

        # Remove unwanted elements
        for pattern in tags_to_remove:
            for elem in root.xpath(pattern):
                elem.getparent().remove(elem)

        # Add declaration and return
        return etree.tostring(root, encoding='unicode', pretty_print=True)

    except Exception as e:
        # If parsing fails, return original
        print(f"Warning: Could not normalize XML: {e}")
        return xml_content


class MusicXMLErrorHandler:
    """Handler for MusicXML parsing errors."""

    @staticmethod
    def validate(xml_content: str) -> tuple[bool, str]:
        """
        Validate MusicXML structure.

        Args:
            xml_content: MusicXML string to validate

        Returns:
            (is_valid, error_message)
        """
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))

            # Check for required MusicXML elements
            if root.tag not in ['score-partwise', 'score-timewise']:
                return False, f"Root element must be score-partwise or score-timewise, got {root.tag}"

            # Check for part-list
            part_list = root.find('.//part-list')
            if part_list is None:
                return False, "Missing part-list element"

            # Check for at least one part
            parts = root.findall('.//part')
            if not parts:
                return False, "No parts found in score"

            return True, "Valid MusicXML"

        except etree.XMLSyntaxError as e:
            return False, f"XML Syntax Error: {e}"
        except Exception as e:
            return False, f"Validation Error: {e}"


if __name__ == "__main__":
    # Example usage
    extractor = MXLExtractor(
        input_dir="data/raw",
        output_dir="data/processed"
    )
    extractor.extract_all()
