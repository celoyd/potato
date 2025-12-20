"""Potato’s custom errors."""


class ImageDimensionError(Exception):
    """Your image is the wrong size."""


class OldSatelliteError(Exception):
    """Your satellite is too old."""


class MissingImageError(Exception):
    """You have only one of pan and mul."""


class TooManyNullsError(Exception):
    """Your image part has too many zeros."""
