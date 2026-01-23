"""
Data processing module for Verglas.

Handles extraction, normalization, and loading of MusicXML files.
"""

from .mxl_extractor import MXLExtractor, normalize_musicxml, MusicXMLErrorHandler
from .musicxml_dataset import MusicXMLDataset, StreamingMusicXMLDataset, create_train_val_split
from .normalizer import preprocess_for_training, clean_xml_whitespace

__all__ = [
    "MXLExtractor",
    "normalize_musicxml",
    "MusicXMLErrorHandler",
    "MusicXMLDataset",
    "StreamingMusicXMLDataset",
    "create_train_val_split",
    "preprocess_for_training",
    "clean_xml_whitespace",
]
