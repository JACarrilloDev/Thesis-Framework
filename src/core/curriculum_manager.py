from enum import Enum
import numpy as np
from .logger import setup_logger

curr_logger = setup_logger('curriculum_logger', 'logs/curriculum.log')

class CurriculumStage(Enum):
    REACH = "reach_training"
    GRASP = "grasp_training"
    CARRY = "carry_training"
    NAVIGATE = "navigation_training"
    PLACE = "place_training"
    FULL_TASK = "full_task"

class CurriculumManager:
    def __init__(self, config: dict):
        self.config = config
        self.current_stage = CurriculumStage.REACH
        self.stage_metrics = {stage: [] for stage in CurriculumStage}
        self.success_threshold = config.get('success_threshold', 0.8)
        self.min_episodes = config.get('min_episodes_per_stage', 100)
        
    def get_stage_config(self) -> dict:
        """Returns configuration specific to current training stage."""
        base_config = self.config.copy()
        
        if self.current_stage == CurriculumStage.REACH:
            # Only reward reaching the projector
            base_config['initial_robot_poses'] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # Simple starting pose
            base_config['initial_projector_poses'] = [[1.0, 0.0, 0.8, 0.0, 0.0, 0.0]]  # Fixed position
            base_config['reward_weights'] = {
                'reaching': 1.0,
                'grasping': 0.0,
                'carrying': 0.0,
                'placing': 0.0
            }
            
        elif self.current_stage == CurriculumStage.GRASP:
            # Robot starts near projector, focus on grasping
            base_config['initial_robot_poses'] = [[0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]
            base_config['initial_projector_poses'] = [[1.0, 0.0, 0.8, 0.0, 0.0, 0.0]]
            base_config['reward_weights'] = {
                'reaching': 0.3,
                'grasping': 1.0,
                'carrying': 0.0,
                'placing': 0.0
            }
            
        elif self.current_stage == CurriculumStage.CARRY:
            # Start with projector grasped, learn to move
            base_config['start_with_projector_grasped'] = True
            base_config['initial_robot_poses'] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            base_config['target_poses'] = [[2.0, 0.0, 0.8, 0.0, 0.0, 0.0]]  # Simple target
            base_config['reward_weights'] = {
                'reaching': 0.0,
                'grasping': 0.2,
                'carrying': 1.0,
                'placing': 0.0
            }
            
        elif self.current_stage == CurriculumStage.NAVIGATE:
            # Add simple obstacles, keep target relatively close
            base_config['start_with_projector_grasped'] = True
            base_config['include_obstacles'] = True
            base_config['reward_weights'] = {
                'reaching': 0.0,
                'grasping': 0.2,
                'carrying': 1.0,
                'placing': 0.0,
                'collision_avoidance': 1.0
            }
            
        elif self.current_stage == CurriculumStage.PLACE:
            # Focus on placing mechanics
            base_config['start_with_projector_grasped'] = True
            base_config['target_poses'] = [[2.0, 0.0, 0.8, 0.0, 0.0, 0.0]]
            base_config['reward_weights'] = {
                'reaching': 0.0,
                'grasping': 0.2,
                'carrying': 0.3,
                'placing': 1.0
            }
            
        else:  # FULL_TASK
            # Use original full task configuration
            pass
            
        return base_config

    def update_stage_metrics(self, stage: CurriculumStage, success_rate: float):
        """Track performance metrics for each stage."""
        self.stage_metrics[stage].append(success_rate)
        
    def should_advance_stage(self) -> bool:
        """Check if ready to advance to next stage."""
        if len(self.stage_metrics[self.current_stage]) < self.min_episodes:
            return False
            
        # Check last N episodes performance
        recent_performance = np.mean(self.stage_metrics[self.current_stage][-20:])
        return recent_performance >= self.success_threshold
        
    def advance_stage(self) -> bool:
        """Try to advance to next stage. Returns True if advanced."""
        if not self.should_advance_stage():
            return False
            
        stages = list(CurriculumStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            curr_logger.info(f"Advanced to stage: {self.current_stage.value}")
            return True
            
        return False  # Already at final stage

    def get_current_stage(self) -> CurriculumStage:
        return self.current_stage