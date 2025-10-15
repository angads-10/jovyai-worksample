"""
API Fine-tuning Pipeline for Decision Transformer

This script converts Decision Transformer sequences to JSONL format
for supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT)
on OpenAI API or similar platforms.

Author: Jovy AI Research Team
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from tqdm import tqdm

from dataset import TrajectoryDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIFineTuneConverter:
    """
    Converts Decision Transformer data to API fine-tuning format.
    
    Supports both SFT (Supervised Fine-Tuning) and RFT (Reinforcement Fine-Tuning)
    formats for OpenAI API and similar platforms.
    """
    
    def __init__(self,
                 max_tokens: int = 2048,
                 token_format: str = 'json'):
        """
        Initialize converter.
        
        Args:
            max_tokens: Maximum tokens per training example
            token_format: Format for token representation ('json' or 'text')
        """
        self.max_tokens = max_tokens
        self.token_format = token_format
    
    def trajectory_to_sft_format(self, 
                                trajectory: Dict,
                                include_rewards: bool = True) -> Dict:
        """
        Convert trajectory to SFT format.
        
        Args:
            trajectory: Trajectory data
            include_rewards: Whether to include reward information
            
        Returns:
            SFT-formatted example
        """
        states = trajectory['states']
        actions = trajectory['actions']
        rtgs = trajectory['rtgs']
        rewards = trajectory.get('rewards', [])
        
        # Create system message
        system_message = """You are a medical AI assistant that learns from hospital trajectories. 
Given a sequence of patient states and return-to-go (RTG) values, predict the next action (medical intervention).
Focus on safe, evidence-based decisions that maximize patient outcomes."""
        
        # Create conversation messages
        messages = [{"role": "system", "content": system_message}]
        
        # Build the trajectory sequence
        trajectory_text = "Hospital Trajectory:\n"
        
        for i in range(len(states)):
            trajectory_text += f"Timestep {i+1}:\n"
            trajectory_text += f"  State: {self._format_state(states[i])}\n"
            trajectory_text += f"  RTG: {rtgs[i]:.3f}\n"
            
            if include_rewards and i < len(rewards):
                trajectory_text += f"  Reward: {rewards[i]:.3f}\n"
            
            trajectory_text += f"  Action: {self._format_action(actions[i])}\n\n"
        
        # Create user message
        user_message = trajectory_text + "\nBased on this trajectory, predict the next action for the final state."
        
        # Create assistant message (target action)
        if len(actions) > 0:
            target_action = self._format_action(actions[-1])
            assistant_message = f"The next action should be: {target_action}"
        else:
            assistant_message = "No action available."
        
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ])
        
        return {
            "messages": messages,
            "trajectory_id": trajectory.get('trajectory_id', 0),
            "length": len(states)
        }
    
    def trajectory_to_rft_format(self,
                                trajectory: Dict,
                                preference_data: Optional[Dict] = None) -> Dict:
        """
        Convert trajectory to RFT format.
        
        Args:
            trajectory: Trajectory data
            preference_data: Optional preference ranking data
            
        Returns:
            RFT-formatted example
        """
        # For RFT, we typically need preference pairs
        # This is a simplified implementation
        
        states = trajectory['states']
        actions = trajectory['actions']
        rtgs = trajectory['rtgs']
        
        # Create prompt
        prompt = self._create_rft_prompt(states, rtgs)
        
        # Create chosen response (higher quality)
        chosen_response = self._format_action(actions[-1]) if len(actions) > 0 else "No action"
        
        # Create rejected response (lower quality - simplified)
        rejected_response = "Conservative observation only"  # Simplified rejection
        
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "trajectory_id": trajectory.get('trajectory_id', 0)
        }
    
    def _format_state(self, state: np.ndarray) -> str:
        """Format state vector for text representation."""
        if self.token_format == 'json':
            return json.dumps(state.tolist())
        else:
            # Create human-readable format
            state_labels = [
                'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
                'respiratory_rate', 'oxygen_saturation', 'age', 'gender',
                'weight', 'height', 'glucose', 'sodium', 'potassium',
                'creatinine', 'bun', 'hemoglobin', 'wbc_count'
            ]
            
            formatted = []
            for i, value in enumerate(state):
                label = state_labels[i] if i < len(state_labels) else f'feature_{i}'
                formatted.append(f"{label}: {value:.2f}")
            
            return ", ".join(formatted)
    
    def _format_action(self, action: np.ndarray) -> str:
        """Format action vector for text representation."""
        if self.token_format == 'json':
            return json.dumps(action.tolist())
        else:
            # Create human-readable format
            action_labels = [
                'fluid_input', 'fluid_output', 'vasopressor_dose',
                'ventilation_mode', 'fio2', 'peep', 'antibiotic_flag',
                'pain_med_flag', 'sedation_flag'
            ]
            
            formatted = []
            for i, value in enumerate(action):
                label = action_labels[i] if i < len(action_labels) else f'action_{i}'
                formatted.append(f"{label}: {value:.2f}")
            
            return ", ".join(formatted)
    
    def _create_rft_prompt(self, states: np.ndarray, rtgs: np.ndarray) -> str:
        """Create prompt for RFT format."""
        prompt = "Given the following patient trajectory, provide the best medical intervention:\n\n"
        
        for i in range(len(states)):
            prompt += f"Timestep {i+1}: State={self._format_state(states[i])}, RTG={rtgs[i]:.3f}\n"
        
        prompt += "\nProvide the optimal action:"
        return prompt
    
    def convert_dataset_to_sft(self,
                              dataset: TrajectoryDataset,
                              output_path: str,
                              max_examples: Optional[int] = None) -> int:
        """
        Convert entire dataset to SFT format.
        
        Args:
            dataset: Trajectory dataset
            output_path: Output JSONL file path
            max_examples: Maximum number of examples to convert
            
        Returns:
            Number of examples converted
        """
        logger.info(f"Converting dataset to SFT format...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        max_examples = max_examples or len(dataset)
        
        with open(output_path, 'w') as f:
            for i in tqdm(range(min(len(dataset), max_examples)), desc="Converting to SFT"):
                try:
                    # Get trajectory from dataset
                    sample = dataset[i]
                    
                    # Convert to trajectory format
                    trajectory = {
                        'states': sample['states'].numpy(),
                        'actions': sample['actions'].numpy(),
                        'rtgs': sample['rtgs'].numpy(),
                        'rewards': sample.get('rewards', np.zeros_like(sample['rtgs'])).numpy(),
                        'trajectory_id': sample.get('trajectory_id', i)
                    }
                    
                    # Convert to SFT format
                    sft_example = self.trajectory_to_sft_format(trajectory)
                    
                    # Write to JSONL
                    f.write(json.dumps(sft_example) + '\n')
                    converted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to convert example {i}: {e}")
                    continue
        
        logger.info(f"Converted {converted_count} examples to SFT format")
        logger.info(f"Output saved to: {output_path}")
        
        return converted_count
    
    def convert_dataset_to_rft(self,
                              dataset: TrajectoryDataset,
                              output_path: str,
                              max_examples: Optional[int] = None) -> int:
        """
        Convert entire dataset to RFT format.
        
        Args:
            dataset: Trajectory dataset
            output_path: Output JSONL file path
            max_examples: Maximum number of examples to convert
            
        Returns:
            Number of examples converted
        """
        logger.info(f"Converting dataset to RFT format...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        max_examples = max_examples or len(dataset)
        
        with open(output_path, 'w') as f:
            for i in tqdm(range(min(len(dataset), max_examples)), desc="Converting to RFT"):
                try:
                    # Get trajectory from dataset
                    sample = dataset[i]
                    
                    # Convert to trajectory format
                    trajectory = {
                        'states': sample['states'].numpy(),
                        'actions': sample['actions'].numpy(),
                        'rtgs': sample['rtgs'].numpy(),
                        'trajectory_id': sample.get('trajectory_id', i)
                    }
                    
                    # Convert to RFT format
                    rft_example = self.trajectory_to_rft_format(trajectory)
                    
                    # Write to JSONL
                    f.write(json.dumps(rft_example) + '\n')
                    converted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to convert example {i}: {e}")
                    continue
        
        logger.info(f"Converted {converted_count} examples to RFT format")
        logger.info(f"Output saved to: {output_path}")
        
        return converted_count
    
    def create_api_upload_script(self,
                               sft_path: str,
                               rft_path: Optional[str] = None,
                               script_path: str = "upload_to_api.py") -> str:
        """
        Create script for uploading data to OpenAI API.
        
        Args:
            sft_path: Path to SFT JSONL file
            rft_path: Path to RFT JSONL file (optional)
            script_path: Output script path
            
        Returns:
            Path to created script
        """
        script_content = f'''"""
Auto-generated script for uploading fine-tuning data to OpenAI API.

Generated by Decision Transformer API Fine-tuning Pipeline.
"""

import openai
import json
from pathlib import Path
import time

# Configure OpenAI API
openai.api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key

def upload_file(file_path: str, purpose: str = "fine-tune") -> str:
    """Upload file to OpenAI."""
    with open(file_path, 'rb') as f:
        response = openai.File.create(
            file=f,
            purpose=purpose
        )
    return response.id

def create_fine_tune_job(training_file_id: str, 
                        validation_file_id: str = None,
                        model: str = "gpt-3.5-turbo",
                        suffix: str = "decision-transformer") -> str:
    """Create fine-tuning job."""
    params = {{
        "training_file": training_file_id,
        "model": model,
        "suffix": suffix
    }}
    
    if validation_file_id:
        params["validation_file"] = validation_file_id
    
    response = openai.FineTuningJob.create(**params)
    return response.id

def main():
    """Main upload script."""
    print("Starting API fine-tuning upload...")
    
    # Upload SFT data
    print("Uploading SFT data...")
    sft_file_id = upload_file("{sft_path}")
    print(f"SFT file uploaded with ID: {{sft_file_id}}")
    
    # Upload RFT data if available
    rft_file_id = None
    if Path("{rft_path}").exists():
        print("Uploading RFT data...")
        rft_file_id = upload_file("{rft_path}")
        print(f"RFT file uploaded with ID: {{rft_file_id}}")
    
    # Create SFT fine-tuning job
    print("Creating SFT fine-tuning job...")
    sft_job_id = create_fine_tune_job(
        training_file_id=sft_file_id,
        suffix="decision-transformer-sft"
    )
    print(f"SFT job created with ID: {{sft_job_id}}")
    
    # Create RFT fine-tuning job if RFT data available
    if rft_file_id:
        print("Creating RFT fine-tuning job...")
        rft_job_id = create_fine_tune_job(
            training_file_id=rft_file_id,
            suffix="decision-transformer-rft"
        )
        print(f"RFT job created with ID: {{rft_job_id}}")
    
    print("Upload complete!")
    print(f"Monitor jobs at: https://platform.openai.com/finetune")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path(script_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Upload script created: {script_path}")
        return str(script_path)

def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(description='Convert Decision Transformer data for API fine-tuning')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to trajectory data')
    parser.add_argument('--output_dir', type=str, default='api_data/',
                       help='Output directory for converted data')
    parser.add_argument('--format', type=str, choices=['sft', 'rft', 'both'], default='both',
                       help='Conversion format')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to convert')
    parser.add_argument('--token_format', type=str, choices=['json', 'text'], default='text',
                       help='Token representation format')
    parser.add_argument('--create_upload_script', action='store_true',
                       help='Create API upload script')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = TrajectoryDataset(args.data_path, max_length=100)
    
    # Create converter
    converter = APIFineTuneConverter(token_format=args.token_format)
    
    # Convert data
    if args.format in ['sft', 'both']:
        sft_path = output_dir / 'sft_data.jsonl'
        sft_count = converter.convert_dataset_to_sft(
            dataset, str(sft_path), args.max_examples
        )
        logger.info(f"SFT conversion complete: {sft_count} examples")
    
    if args.format in ['rft', 'both']:
        rft_path = output_dir / 'rft_data.jsonl'
        rft_count = converter.convert_dataset_to_rft(
            dataset, str(rft_path), args.max_examples
        )
        logger.info(f"RFT conversion complete: {rft_count} examples")
    
    # Create upload script if requested
    if args.create_upload_script:
        sft_path = output_dir / 'sft_data.jsonl'
        rft_path = output_dir / 'rft_data.jsonl' if args.format in ['rft', 'both'] else None
        
        script_path = converter.create_api_upload_script(
            str(sft_path), 
            str(rft_path) if rft_path else None,
            str(output_dir / 'upload_to_api.py')
        )
        
        logger.info(f"Upload script created: {script_path}")
        logger.info("Remember to:")
        logger.info("1. Replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key")
        logger.info("2. Install openai package: pip install openai")
        logger.info("3. Run the upload script: python upload_to_api.py")

if __name__ == "__main__":
    main()
