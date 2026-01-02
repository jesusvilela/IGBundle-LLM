
"""
mfr_utils.py - Model-First Reasoning Utilities

This module implements the prompting strategy for Model-First Reasoning (MFR),
separating problem solving into:
1. Model Construction: Explicitly defining entities, states, actions, and constraints.
2. Reasoning: Generating a solution strictly based on the constructed model.
"""

def construct_phase1_prompt(task_description):
    """
    Constructs the prompt for Phase 1: Model Construction.
    Uses Alpaca template to match the fine-tuned adapter distribution.
    """
    # Clean trailing triggers from the raw task description
    clean_task = task_description.strip()
    if clean_task.endswith("Output:"):
        clean_task = clean_task[:-7].strip()

    instruction = (
        "You are an expert systems modeler. Before solving this puzzle, you must explicitly construct a structural model of the problem.\n"
        "Analyze the input grids and examples. Define the following components:\n"
        "1. **ENTITIES**: List the objects involved (e.g., colored blocks, lines, shapes). Describe their properties.\n"
        "2. **STATE VARIABLES**: What properties change? (e.g., position moves, color changes).\n"
        "3. **ACTIONS/TRANSFORMATIONS**: What operations transforms Input to Output?\n"
        "4. **CONSTRAINTS**: What rules must always hold?\n\n"
        "**OUTPUT FORMAT**:\n"
        "Provide your model in a structured format. DO NOT generate the final output grid yet. Only define the model.\n"
        "Begin your response with '## PROBLEM MODEL'."
    )
    
    # Alpaca Template
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{clean_task}\n\n"
        "### Response:\n"
    )
    return prompt

def construct_phase2_prompt(task_description, model_output):
    """
    Constructs the prompt for Phase 2: Reasoning.
    Uses Alpaca template.
    """
    # Clean trailing triggers
    clean_task = task_description.strip()
    if clean_task.endswith("Output:"):
        clean_task = clean_task[:-7].strip()

    instruction = (
        "Using ONLY the explicitly defined entities, actions, and constraints in the Problem Model provided below, solve the Test case.\n"
        "1. Apply the transformations step-by-step.\n"
        "2. Verify that all constraints are respected.\n"
        "3. Generate the Final Output Grid."
    )
    
    # Combine Task + Model into Input context
    full_input = (
        f"{clean_task}\n\n"
        "--- EXPLICIT PROBLEM MODEL ---\n"
        f"{model_output}"
    )
    
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{full_input}\n\n"
        "### Response:\n"
        "Output:"
    )
    return prompt
