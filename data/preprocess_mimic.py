"""
MIMIC-III Dataset Preprocessing for Offline RL Decision Transformer

This script processes MIMIC-III ICU data to create hospital trajectories
suitable for training Decision Transformers in offline RL settings.

Author: Jovy AI Research Team
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIMICProcessor:
    """
    Processes MIMIC-III data into offline RL trajectories.
    
    Each trajectory consists of:
    - States: Vitals, demographics, lab values
    - Actions: Interventions (fluids, ventilation, medications)
    - Rewards: +1 (survival/discharge), -1 (mortality)
    - RTG: Return-to-Go (cumulative future reward)
    """
    
    def __init__(self, data_dir: str = "data/mimic/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State features (normalized vitals and demographics)
        self.state_features = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
            'respiratory_rate', 'oxygen_saturation', 'age', 'gender',
            'weight', 'height', 'glucose', 'sodium', 'potassium',
            'creatinine', 'bun', 'hemoglobin', 'wbc_count'
        ]
        
        # Action features (interventions)
        self.action_features = [
            'fluid_input', 'fluid_output', 'vasopressor_dose',
            'ventilation_mode', 'fio2', 'peep', 'antibiotic_flag',
            'pain_med_flag', 'sedation_flag'
        ]
        
    def load_mimic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load MIMIC-III tables from CSV files.
        
        Note: In practice, you would download from Kaggle:
        https://www.kaggle.com/datasets/asjad99/mimiciii
        
        Returns:
            Dictionary of DataFrames for each MIMIC table
        """
        logger.info("Loading MIMIC-III data...")
        
        # For demo purposes, create synthetic data that mimics MIMIC structure
        # In practice, load from actual MIMIC files
        if not self._check_data_exists():
            logger.warning("MIMIC data not found. Creating synthetic data for demo...")
            return self._create_synthetic_data()
        
        data = {}
        tables = ['patients', 'admissions', 'icustays', 'chartevents', 'labevents']
        
        for table in tables:
            file_path = self.data_dir / f"{table}.csv"
            if file_path.exists():
                data[table] = pd.read_csv(file_path)
                logger.info(f"Loaded {table}: {len(data[table])} records")
            else:
                logger.warning(f"Table {table} not found at {file_path}")
        
        return data
    
    def _check_data_exists(self) -> bool:
        """Check if actual MIMIC data files exist."""
        required_files = ['patients.csv', 'admissions.csv', 'icustays.csv']
        return all((self.data_dir / f).exists() for f in required_files)
    
    def _create_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Create synthetic MIMIC-like data for demonstration."""
        np.random.seed(42)
        n_patients = 1000
        n_admissions = 1200
        n_icustays = 800
        
        # Synthetic patients
        patients = pd.DataFrame({
            'subject_id': range(1, n_patients + 1),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'anchor_age': np.random.randint(18, 90, n_patients),
            'anchor_year': np.random.randint(2010, 2020, n_patients),
            'dod': np.random.choice([None, '2020-01-01'], n_patients, p=[0.8, 0.2])
        })
        
        # Synthetic admissions
        admissions = pd.DataFrame({
            'subject_id': np.random.choice(patients['subject_id'], n_admissions),
            'hadm_id': range(1, n_admissions + 1),
            'admittime': pd.date_range('2010-01-01', periods=n_admissions, freq='1H'),
            'dischtime': pd.date_range('2010-01-02', periods=n_admissions, freq='1H'),
            'deathtime': [None] * n_admissions,
            'admission_type': np.random.choice(['EMERGENCY', 'ELECTIVE', 'URGENT'], n_admissions),
            'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private'], n_admissions)
        })
        
        # Synthetic ICU stays
        icustays = pd.DataFrame({
            'subject_id': np.random.choice(patients['subject_id'], n_icustays),
            'hadm_id': np.random.choice(admissions['hadm_id'], n_icustays),
            'icustay_id': range(1, n_icustays + 1),
            'intime': pd.date_range('2010-01-01', periods=n_icustays, freq='2H'),
            'outtime': pd.date_range('2010-01-03', periods=n_icustays, freq='2H'),
            'first_careunit': np.random.choice(['MICU', 'SICU', 'CCU', 'TSICU'], n_icustays)
        })
        
        # Synthetic chart events (vitals)
        chartevents = []
        for icu_id in icustays['icustay_id']:
            n_events = np.random.randint(50, 200)
            for i in range(n_events):
                chartevents.append({
                    'subject_id': icustays[icustays['icustay_id'] == icu_id]['subject_id'].iloc[0],
                    'hadm_id': icustays[icustays['icustay_id'] == icu_id]['hadm_id'].iloc[0],
                    'icustay_id': icu_id,
                    'itemid': np.random.choice([211, 51, 442, 225312, 618, 220045, 223761]),
                    'charttime': pd.Timestamp('2010-01-01') + pd.Timedelta(hours=i*2),
                    'valuenum': np.random.normal(80, 20)
                })
        
        chartevents = pd.DataFrame(chartevents)
        
        return {
            'patients': patients,
            'admissions': admissions, 
            'icustays': icustays,
            'chartevents': chartevents
        }
    
    def extract_trajectories(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Extract hospital trajectories from MIMIC data.
        
        Each trajectory represents one ICU stay with:
        - States: Normalized vitals and demographics
        - Actions: Interventions and treatments
        - Rewards: Based on survival/discharge outcome
        
        Returns:
            List of trajectory dictionaries
        """
        logger.info("Extracting hospital trajectories...")
        
        trajectories = []
        icustays = data['icustays']
        
        for _, stay in icustays.iterrows():
            try:
                trajectory = self._extract_single_trajectory(stay, data)
                if trajectory and len(trajectory['states']) > 5:  # Minimum trajectory length
                    trajectories.append(trajectory)
            except Exception as e:
                logger.warning(f"Failed to extract trajectory for ICU stay {stay['icustay_id']}: {e}")
                continue
        
        logger.info(f"Extracted {len(trajectories)} valid trajectories")
        return trajectories
    
    def _extract_single_trajectory(self, stay: pd.Series, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Extract a single ICU trajectory."""
        icu_id = stay['icustay_id']
        subject_id = stay['subject_id']
        
        # Get patient demographics
        patient = data['patients'][data['patients']['subject_id'] == subject_id].iloc[0]
        
        # Get chart events (vitals) for this ICU stay
        chart_events = data['chartevents'][data['chartevents']['icustay_id'] == icu_id]
        
        if len(chart_events) < 5:
            return None
        
        # Sort by time
        chart_events = chart_events.sort_values('charttime')
        
        # Create states (normalized vitals + demographics)
        states = []
        actions = []
        
        for _, event in chart_events.iterrows():
            # Create state vector
            state = self._create_state_vector(event, patient)
            states.append(state)
            
            # Create action vector (simplified for demo)
            action = self._create_action_vector(event)
            actions.append(action)
        
        # Determine reward based on outcome
        # In practice, this would use actual mortality/discharge data
        reward = self._calculate_reward(stay, patient)
        
        # Create Return-to-Go (RTG) sequence
        rtgs = self._calculate_rtg_sequence(reward, len(states))
        
        return {
            'icustay_id': icu_id,
            'subject_id': subject_id,
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array([reward] * len(states)),
            'rtgs': np.array(rtgs),
            'length': len(states)
        }
    
    def _create_state_vector(self, event: pd.Series, patient: pd.Series) -> np.ndarray:
        """Create normalized state vector from vitals and demographics."""
        # Simplified state creation for demo
        # In practice, would map MIMIC itemids to specific vitals
        state = np.random.normal(0, 1, len(self.state_features))
        
        # Add some patient-specific information
        state[6] = (patient['anchor_age'] - 50) / 30  # Normalized age
        state[7] = 1 if patient['gender'] == 'M' else -1  # Gender encoding
        
        return state
    
    def _create_action_vector(self, event: pd.Series) -> np.ndarray:
        """Create action vector from interventions."""
        # Simplified action creation for demo
        action = np.random.normal(0, 1, len(self.action_features))
        
        # Add some event-specific actions
        if event['itemid'] == 211:  # Heart rate
            action[0] = np.random.uniform(0, 1000) / 1000  # Fluid input
        elif event['itemid'] == 51:  # Systolic BP
            action[1] = np.random.uniform(0, 500) / 500  # Fluid output
        
        return action
    
    def _calculate_reward(self, stay: pd.Series, patient: pd.Series) -> float:
        """Calculate reward based on patient outcome."""
        # Simplified reward calculation
        # In practice, would use actual mortality data
        if pd.isna(patient['dod']):
            return 1.0  # Survival
        else:
            return -1.0  # Mortality
    
    def _calculate_rtg_sequence(self, final_reward: float, length: int) -> List[float]:
        """Calculate Return-to-Go sequence."""
        # Simple RTG calculation: final reward discounted backwards
        rtgs = []
        for i in range(length):
            rtg = final_reward * (0.99 ** (length - 1 - i))
            rtgs.append(rtg)
        return rtgs
    
    def save_trajectories(self, trajectories: List[Dict], output_file: str = "hospital_trajectories.csv"):
        """Save trajectories to CSV file."""
        logger.info(f"Saving {len(trajectories)} trajectories to {output_file}")
        
        # Flatten trajectories for CSV storage
        flattened_data = []
        
        for traj in trajectories:
            for i in range(len(traj['states'])):
                flattened_data.append({
                    'icustay_id': traj['icustay_id'],
                    'subject_id': traj['subject_id'],
                    'timestep': i,
                    'state': traj['states'][i].tolist(),
                    'action': traj['actions'][i].tolist(),
                    'reward': traj['rewards'][i],
                    'rtg': traj['rtgs'][i],
                    'trajectory_length': traj['length']
                })
        
        df = pd.DataFrame(flattened_data)
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Saved trajectories to {output_path}")
        
        return output_path

def main():
    """Main preprocessing pipeline."""
    processor = MIMICProcessor()
    
    # Load data
    data = processor.load_mimic_data()
    
    # Extract trajectories
    trajectories = processor.extract_trajectories(data)
    
    # Save trajectories
    output_path = processor.save_trajectories(trajectories)
    
    logger.info(f"Preprocessing complete! Output saved to: {output_path}")
    logger.info(f"Total trajectories: {len(trajectories)}")
    
    # Print sample statistics
    if trajectories:
        lengths = [t['length'] for t in trajectories]
        logger.info(f"Average trajectory length: {np.mean(lengths):.2f}")
        logger.info(f"Trajectory length range: {min(lengths)} - {max(lengths)}")

if __name__ == "__main__":
    main()
