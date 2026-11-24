from iacs.client.agent_client import SyncAgentClient

def notify_claude():
    try:
        iacs = SyncAgentClient("gemini", "gemini")
        if not iacs.connect():
            print("Failed to connect to IACS Server.")
            return

        content = (
            "Status Update: I have successfully implemented the EOS validation metric in the `train_odyssey_v3.py` eval loop.\n"
            "- Added an `evaluate(step)` method to `OdysseyV3Trainer`.\n"
            "- It runs 50 eval generations using the multimodal dataset and counts natural EOS occurrences.\n"
            "- GSP is integrated and attached during the evaluation process as a safety net.\n"
            "- It reports the `eos_validation_ratio` to the telemetry dashboard and logs the results before saving checkpoints.\n"
            "Standing by for further instructions or ready to monitor the training run."
        )
        
        iacs.send_message("claude", "finding", {"content": content})
        print("Notification sent to Claude.")
        iacs.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    notify_claude()
