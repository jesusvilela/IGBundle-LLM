import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class TestPackageExports(unittest.TestCase):
    def test_epic_packages_export_core_symbols(self):
        from igbundle.modules import ManifoldDecompiler, SheafConsensus, VisionProjector
        from igbundle.fibers import ConstraintExtractor, FiberExecutor
        from igbundle.cognition import MetaCognitiveLoop
        from igbundle.dynamics import HamiltonianSystem, TimeReversalCheck

        self.assertIsNotNone(ManifoldDecompiler)
        self.assertIsNotNone(SheafConsensus)
        self.assertIsNotNone(VisionProjector)
        self.assertIsNotNone(ConstraintExtractor)
        self.assertIsNotNone(FiberExecutor)
        self.assertIsNotNone(MetaCognitiveLoop)
        self.assertIsNotNone(HamiltonianSystem)
        self.assertIsNotNone(TimeReversalCheck)


if __name__ == "__main__":
    unittest.main()