"""
MusicXML Normalizer

Normalizes MusicXML files for training by:
1. Removing layout/formatting metadata
2. Standardizing musical element representations
3. Adding special tokens for model training
"""

import re
from lxml import etree
from typing import Optional


def normalize_musicxml(xml_content: str, keep_layout: bool = False) -> str:
    """
    Normalize MusicXML content for training.

    Args:
        xml_content: Raw MusicXML string
        keep_layout: If True, preserve layout information

    Returns:
        Normalized MusicXML string
    """
    try:
        # Parse XML
        root = etree.fromstring(xml_content.encode('utf-8'))

        # Remove layout-specific elements if requested
        if not keep_layout:
            _remove_layout_elements(root)

        # Normalize musical elements
        _normalize_note_elements(root)
        _normalize_measure_elements(root)

        # Convert back to string
        result = etree.tostring(root, encoding='unicode', pretty_print=True)

        return result

    except Exception as e:
        # If parsing fails, return original with a warning
        print(f"Warning: Could not normalize XML: {e}")
        return xml_content


def _remove_layout_elements(root: etree.Element) -> None:
    """Remove layout and formatting elements from MusicXML tree."""

    # Elements to remove (layout, formatting, appearance)
    tags_to_remove = [
        'page-layout', 'system-layout', 'system-margins',
        'scaling', 'millimeters', 'tenths',
        'appearance', 'line-width', 'note-size', 'distance',
        'print', 'page-height', 'page-width', 'page-margins',
        'credit', 'credit-words', 'credit-type',
        'word-font', 'lyric-font',
        'defaults',  # Remove defaults section
    ]

    for tag in tags_to_remove:
        for elem in root.iter(f'{{{http://www.musicxml.org/ns/lilypond}}}{tag}'):
            elem.getparent().remove(elem)
        for elem in root.iter(f'{{{http://www.musicxml.org/ns/lilypond}}}{tag}'):
            elem.getparent().remove(elem)
        for elem in root.findall(f'.//{tag}'):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)


def _normalize_note_elements(root: etree.Element) -> None:
    """Normalize note elements for consistency."""

    for note in root.findall('.//note'):
        # Ensure all notes have a duration element
        if note.find('duration') is None:
            duration = etree.SubElement(note, 'duration')
            duration.text = '1'

        # Normalize rest elements
        if note.find('rest') is not None:
            rest = note.find('rest')
            if rest.get('measure') == 'yes':
                rest.set('measure', 'yes')


def _normalize_measure_elements(root: etree.Element) -> None:
    """Normalize measure elements for consistency."""

    for measure in root.findall('.//measure'):
        # Ensure all measures have a number attribute
        if measure.get('number') is None:
            measure.set('number', '1')

        # Remove implicit='yes' from first measure
        if measure.get('number') == '1' and measure.get('implicit') == 'yes':
            del measure.attrib['implicit']


def strip_xml_header(xml_content: str) -> str:
    """Strip XML declaration from content."""
    return re.sub(r'<\?xml[^?]*\?>', '', xml_content).strip()


def add_special_tokens(xml_content: str, bos: str = "<|startofmusic|>", eos: str = "<|endofmusic|>") -> str:
    """Add special tokens to MusicXML content."""
    return f"{bos}\n{xml_content}\n{eos}"


def truncate_to_musical_content(xml_content: str, max_tokens: int = 4000) -> str:
    """
    Truncate MusicXML to focus on musical content.

    Keeps measures and parts while trimming metadata.

    Args:
        xml_content: Full MusicXML string
        max_tokens: Approximate token limit

    Returns:
        Truncated MusicXML string
    """
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))

        # Remove large metadata sections
        for elem in root.findall('.//credit'):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)

        # Truncate to first N measures if too long
        parts = root.findall('.//part')
        for part in parts:
            measures = part.findall('measure')
            if len(measures) > 32:  # Keep ~32 measures max
                for measure in measures[32:]:
                    part.remove(measure)

        return etree.tostring(root, encoding='unicode', pretty_print=True)

    except Exception:
        return xml_content


def clean_xml_whitespace(xml_content: str) -> str:
    """Clean up whitespace in XML while preserving structure."""
    # Remove excessive newlines but preserve structure
    xml_content = re.sub(r'\n\s*\n\s*\n', '\n\n', xml_content)
    # Clean up indentation
    xml_content = re.sub(r'[ \t]+', ' ', xml_content)
    return xml_content.strip()


# Convenience function combining all normalization steps
def preprocess_for_training(
    xml_content: str,
    add_special: bool = True,
    bos: str = "<|startofmusic|>",
    eos: str = "<|endofmusic|>"
) -> str:
    """
    Complete preprocessing pipeline for training data.

    Args:
        xml_content: Raw MusicXML string
        add_special: Whether to add special tokens
        bos: Beginning of sequence token
        eos: End of sequence token

    Returns:
        Processed MusicXML string ready for tokenization
    """
    # Step 1: Normalize structure
    content = normalize_musicxml(xml_content)

    # Step 2: Truncate if needed
    content = truncate_to_musical_content(content)

    # Step 3: Clean whitespace
    content = clean_xml_whitespace(content)

    # Step 4: Add special tokens
    if add_special:
        content = add_special_tokens(content, bos=bos, eos=eos)

    return content
