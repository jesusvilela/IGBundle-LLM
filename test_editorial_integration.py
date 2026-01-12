import unittest
from unittest.mock import MagicMock, patch
import asyncio
import os
import sys

# Add current dir to path to import auxiliary_crew
sys.path.append(os.getcwd())

# Mock the EditorialBoard before importing auxiliary_crew
sys.modules["editorial_board.workflow"] = MagicMock()
from auxiliary_crew import Supervisor, EditorialBoard

class TestEditorialIntegration(unittest.TestCase):
    def setUp(self):
        self.supervisor = Supervisor("TestSuper", "Supervisor")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_run_development_phase_triggers_editorial(self, mock_exists, mock_run):
        # Setup mocks
        mock_exists.return_value = True # Pretend PDF exists
        mock_board_instance = MagicMock()
        
        # We need to mock the EditorialBoard class to return our instance
        with patch("auxiliary_crew.EditorialBoard", return_value=mock_board_instance) as MockBoard:
            # Run the async method
            asyncio.run(self.supervisor.run_development_phase())
            
            # Verify generate_thesis.py was called
            mock_run.assert_any_call(["python", "generate_thesis.py"], check=True)
            
            # Verify EditorialBoard was instantiated
            MockBoard.assert_called_once()
            
            # Verify process_manuscript was called with correct path
            expected_pdf = "output/thesis/IGBundle_Thesis.pdf"
            mock_board_instance.process_manuscript.assert_called_with(expected_pdf)

if __name__ == "__main__":
    unittest.main()
