"""Verify that all tutorial modules can be imported without errors."""

import tutorials.basic_operations.flip  # noqa: F401
import tutorials.basic_operations.readimg  # noqa: F401
import tutorials.basic_operations.resize  # noqa: F401
import tutorials.color_adjustments.imbright  # noqa: F401
import tutorials.color_adjustments.imcontrast  # noqa: F401
import tutorials.color_adjustments.imsaturation  # noqa: F401
import tutorials.edge_detection.cnn_edgedetection  # noqa: F401
import tutorials.edge_detection.edgedetection  # noqa: F401
import tutorials.edge_detection.linesegments  # noqa: F401
import tutorials.edge_detection.lsdlines  # noqa: F401
import tutorials.feature_detection.face_detection  # noqa: F401
import tutorials.feature_detection.gradients  # noqa: F401
import tutorials.feature_detection.houghlines  # noqa: F401
import tutorials.filtering.blur  # noqa: F401
import tutorials.filtering.unsharp  # noqa: F401


def test_imports() -> None:
    pass
