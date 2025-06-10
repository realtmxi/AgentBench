import json
import os
from typing import List, Dict, Any
from src.server.task import Task, Session
from src.typings import (
    SampleIndex, 
    TaskSampleExecutionResult, 
    SampleStatus,
    AgentOutputStatus
)
from src.typings.output import TaskOutput

# Import your existing components
from .agentenv_gaia.environment import GaiaEnvServer
from .agentenv_gaia.load_data import load_gaia_data

class GAIATask(Task):
    def __init__(self, 
                 dataset_type: str = "validation",
                 level: str = "level1", 
                 data_dir: str = "data/",
                 max_rounds: int = 20,
                 **configs):
        super().__init__(**configs)
        
        self.dataset_type = dataset_type
        self.level = level
        self.data_dir = data_dir
        self.max_rounds = max_rounds
        
        # Initialize GAIA environment server
        self.gaia_server = GaiaEnvServer()
        
        # Load dataset
        self.dataset = load_gaia_data(
            data_dir=data_dir,
            level=level,
            dataset=dataset_type
        )
        
    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.dataset)))
    
    async def start_sample(self, index: SampleIndex, session: Session) -> TaskSampleExecutionResult:
        # Create GAIA environment instance
        env_id = self.gaia_server.create(
            id=index,
            dataset_type=self.dataset_type
        )
        
        try:
            # Get initial observation/question
            obs = self.gaia_server.observation(env_id)
            
            # Present the task to agent
            session.inject({"role": "user", "content": obs})
            
            # Multi-turn interaction
            for round_num in range(self.max_rounds):
                # Get agent response
                response = await session.action()
                
                if response.status != AgentOutputStatus.NORMAL:
                    break
                
                # Execute action in GAIA environment
                step_result = self.gaia_server.step(env_id, response.content)
                
                # Check if task is complete
                if step_result.get("done", False):
                    break
                
                # Provide feedback to agent
                if "observation" in step_result:
                    session.inject({
                        "role": "user", 
                        "content": step_result["observation"]
                    })
            
            # Get final result
            final_result = self.gaia_server.observation(env_id)
            
            # Evaluate answer
            correct_answer = self.dataset.iloc[index]["true_answer"]
            agent_answer = self._extract_final_answer(final_result)
            
            score = 1.0 if self._evaluate_answer(agent_answer, correct_answer) else 0.0
            
            return TaskSampleExecutionResult(
                status=SampleStatus.COMPLETED,
                result={
                    "score": score,
                    "agent_answer": agent_answer,
                    "correct_answer": correct_answer,
                    "question": self.dataset.iloc[index]["question"]
                }
            )
            
        finally:
            # Clean up environment
            if env_id:
                # Add cleanup method to GaiaEnvServer if needed
                pass
    
    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        scores = [r.result.get("score", 0.0) for r in results if r.result]
        return {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "total_samples": len(results),
            "correct_answers": sum(scores)
        } 
    
    def _extract_final_answer(self, result):
        # Extract final answer from GAIA environment result
        # Implement based on your result format
        pass
        
    def _evaluate_answer(self, agent_answer, correct_answer):
        # Implement GAIA's answer evaluation logic
        # Handle exact match with normalization
        pass