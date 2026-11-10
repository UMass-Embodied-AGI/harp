"""Dataset format converters, loaded only by their selected entry point.

Public command paths remain in ``scripts/`` for compatibility. Each converter
keeps its existing optional environment requirements; importing this package
does not import TensorFlow, LeRobot, or any baseline model.
"""
