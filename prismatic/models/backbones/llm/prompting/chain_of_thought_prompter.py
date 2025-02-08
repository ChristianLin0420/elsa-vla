"""
chain_of_thought_prompter.py

Defines a PromptBuilder for building Chain of Thought prompts that encourage step-by-step reasoning.
The number of reasoning steps can be configured, and each step has a specific focus in the reasoning chain.
"""

from typing import Optional, List, Dict

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompts for different model families
SYS_PROMPTS = {
    "prismatic": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant breaks down complex tasks into clear, logical steps before providing a final answer. "
        "Each response follows a chain of thought that explains the reasoning process."
    ),
    "openvla": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant analyzes robotic tasks step by step, considering the scene, goals, and required actions. "
        "Each response follows a structured reasoning process before determining the final actions."
    ),
}

# Default reasoning steps for different domains
DEFAULT_REASONING_STEPS = {
    "robotic": [
        "Scene Analysis: Identify and locate key objects, tools, and environmental elements.",
        "Goal Understanding: Determine the target state and what needs to change.",
        "Constraint Identification: Consider physical limitations and safety requirements.",
        "Action Planning: Break down the task into specific, executable actions.",
        "Validation: Verify the planned actions against goals and constraints."
    ],
    "general": [
        "Problem Analysis: Break down the key components of the task.",
        "Context Consideration: Identify relevant information and constraints.",
        "Solution Strategy: Develop an approach to solve the problem.",
        "Step Planning: Define specific steps to implement the solution.",
        "Verification: Check if the solution meets all requirements."
    ]
}

class ChainOfThoughtPromptBuilder(PromptBuilder):
    def __init__(
        self, 
        model_family: str = "openvla", 
        system_prompt: Optional[str] = None,
        domain: str = "robotic",
        num_reasoning_steps: Optional[int] = 4,
        custom_reasoning_steps: Optional[List[str]] = None
    ) -> None:
        """Initialize the Chain of Thought Prompt Builder.
        
        Args:
            model_family: The family of models being used (e.g., "prismatic", "openvla")
            system_prompt: Optional custom system prompt
            domain: Domain for reasoning steps ("robotic" or "general")
            num_reasoning_steps: Number of reasoning steps to use (if None, uses all steps)
            custom_reasoning_steps: Optional list of custom reasoning step descriptions
        """
        super().__init__(model_family, system_prompt)
        self.system_prompt = (SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt).strip() + " "
        
        # Set up reasoning steps
        if custom_reasoning_steps is not None:
            self.reasoning_steps = custom_reasoning_steps
        else:
            all_steps = DEFAULT_REASONING_STEPS[domain]
            if num_reasoning_steps is not None:
                self.reasoning_steps = all_steps[:num_reasoning_steps]
            else:
                self.reasoning_steps = all_steps
        
        # LLaMA-specific tokens
        self.bos, self.eos = "<s>", "</s>"
        
        # Format templates
        self.wrap_human = lambda msg: f"USER: {msg}\nLet's solve this step by step:\n"
        self.wrap_step = lambda step, content: f"{step}:\n{content}\n"
        self.wrap_gpt = lambda steps, conclusion: (
            f"ASSISTANT: I'll help you with that. Let's break it down:\n\n"
            f"{steps}\n"
            f"Based on this reasoning, here's what we should do: {conclusion}{self.eos}"
        )
        
        # Initialize prompt building
        self.prompt, self.turn_count = "", 0
        self.current_reasoning = []

    def add_turn(self, role: str, message: str) -> str:
        """Add a turn to the conversation with chain of thought reasoning.
        
        For assistant responses, structures the output with explicit reasoning steps.
        """
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special handling for first turn
        if self.turn_count == 0:
            wrapped_message = self.system_prompt + self.wrap_human(message)
        elif (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            # For assistant turns, structure the response with reasoning steps
            if isinstance(message, dict):
                # If message is already structured with reasoning
                steps_content = ""
                for step, content in zip(self.reasoning_steps, message["reasoning"]):
                    steps_content += self.wrap_step(step, content)
                wrapped_message = self.wrap_gpt(steps_content, message["conclusion"])
            else:
                # If message is raw text, try to parse it into reasoning steps
                parts = message.split("\n")
                steps_content = ""
                conclusion = parts[-1] if parts else message
                
                # Try to match parts to reasoning steps
                for i, step in enumerate(self.reasoning_steps):
                    content = parts[i] if i < len(parts) else "..."
                    steps_content += self.wrap_step(step, content)
                
                wrapped_message = self.wrap_gpt(steps_content, conclusion)

        # Update prompt
        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        """Get the potential prompt that would result from adding the message."""
        prompt_copy = str(self.prompt)
        
        if self.turn_count == 0:
            wrapped_message = self.system_prompt + self.wrap_human(message)
        else:
            wrapped_message = self.wrap_human(message)
            
        prompt_copy += wrapped_message
        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        """Get the current prompt."""
        return self.prompt.removeprefix(self.bos).rstrip()

    def get_reasoning_steps(self) -> List[str]:
        """Get the current reasoning steps being used."""
        return self.reasoning_steps

    def set_reasoning_steps(self, steps: List[str]) -> None:
        """Update the reasoning steps."""
        self.reasoning_steps = steps
