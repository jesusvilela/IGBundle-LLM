import re

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Checks if the generated completion matches the ground truth answer (GSM8K style).
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion has a specific format.
    e.g. <reasoning>...</reasoning> <answer>...</answer>
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion has a specific format.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion has a specific format.
    """
    responses = [completion[0]["content"] for completion in completions]
    def count_xml(text):
        count = 0.0
        if "<reasoning>" in text:
            count += 0.125
        if "</reasoning>" in text:
            count += 0.125
        if "<answer>" in text:
            count += 0.125
        if "</answer>" in text:
            count += 0.125
        return count
    return [count_xml(r) for r in responses]

def extract_xml_answer(text: str) -> str:
    """
    Extracts the answer from within <answer> tags.
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def soft_energy_reward_func(completions, **kwargs) -> list[float]:
    """
    Approximates 'Hamiltonian Energy' based on output complexity.
    
    Assumption: 'Least Action' principle in reasoning implies:
    1. Conciseness (lower token count for same result).
    2. Structural coherence (fewer formatting breaks).
    
    Reward = - (Length Penalty + Formatting Chaos)
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        # Length penalty (we want efficient reasoning, not verbose yapping)
        # Normalize: target length ~500 chars?
        L = len(text)
        length_penalty = max(0, (L - 2000) / 1000.0) # Penalty starts after 2000 chars
        
        # Chaos penalty: distinct XML tags mismatch?
        # Simple proxy: count of newlines vs content?
        
        # Reward: 1.0 (baseline) - length_penalty
        rewards.append(max(0.0, 1.0 - length_penalty * 0.5))
    return rewards
