#!/usr/bin/env python
"""
DFS Explorer for ALFWorld (AgentBench version)

This script explores ALFWorld game instances and builds a knowledge base
of optimal paths for each game.

Two modes:
1. Expert mode (fast): Use the built-in handcoded expert to get optimal paths
2. BFS mode (slow but complete): Explore all states using BFS

Usage:
    # From AgentBench root directory:
    python -m src.server.tasks.alfworld.dfs_explore --game_file <path_to_game.tw-pddl>
    python -m src.server.tasks.alfworld.dfs_explore --data_path /path/to/data --output knowledge_base.json
    
    # Using the dev.json file list:
    python -m src.server.tasks.alfworld.dfs_explore --split dev --output knowledge_base_dev.json
"""

import os
import sys
import json
import argparse
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert, AlfredExpertType


def load_config(config_file: str) -> dict:
    """Load YAML config file."""
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config


@dataclass 
class ExplorationResult:
    """Result of exploring a single game."""
    game_file: str
    success: bool
    optimal_path: List[str]
    path_length: int
    observations: List[str]  # Observations at each step
    error: Optional[str] = None


class ALFWorldExplorer:
    """Explores ALFWorld games using the handcoded expert or BFS."""
    
    def __init__(self, max_steps: int = 50, use_expert: bool = True):
        self.max_steps = max_steps
        self.use_expert = use_expert
        
    def _create_env(self, game_file: str, with_expert: bool = False):
        """Create a TextWorld environment for the given game file."""
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            facts=True,
            extras=["gamefile"]
        )
        
        wrappers = [AlfredDemangler(), AlfredInfos()]
        
        if with_expert:
            wrappers.append(AlfredExpert(expert_type=AlfredExpertType.HANDCODED))
            request_infos.extras.append("expert_plan")
        
        env_id = textworld.gym.register_game(
            game_file,
            request_infos,
            max_episode_steps=self.max_steps * 2,
            wrappers=wrappers
        )
        
        return textworld.gym.make(env_id)
    
    def explore_with_expert(self, game_file: str, verbose: bool = False) -> ExplorationResult:
        """
        Use the handcoded expert to find the optimal path.
        This is fast because the expert already knows the solution.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Exploring with expert: {game_file}")
            print(f"{'='*60}")
        
        try:
            env = self._create_env(game_file, with_expert=True)
        except Exception as e:
            return ExplorationResult(
                game_file=game_file,
                success=False,
                optimal_path=[],
                path_length=0,
                observations=[],
                error=f"Failed to create env: {str(e)}"
            )
        
        try:
            obs, infos = env.reset()
            
            if verbose:
                print(f"Initial observation:\n{obs[:300]}...")
            
            path = []
            observations = [obs]
            done = False
            steps = 0
            
            while not done and steps < self.max_steps:
                # Get expert action
                expert_plan = infos.get("extra.expert_plan", [])
                if not expert_plan:
                    # Fallback: try "look" or first admissible command
                    admissible = infos.get("admissible_commands", [])
                    if "look" in admissible:
                        action = "look"
                    elif admissible:
                        action = admissible[0]
                    else:
                        break
                else:
                    action = expert_plan[0]
                
                if verbose:
                    print(f"  Step {steps+1}: {action}")
                
                obs, reward, done, infos = env.step(action)
                path.append(action)
                observations.append(obs)
                steps += 1
                
                if infos.get("won", False):
                    done = True
                    if verbose:
                        print(f"  Won after {steps} steps!")
            
            env.close()
            
            success = infos.get("won", False)
            
            return ExplorationResult(
                game_file=game_file,
                success=success,
                optimal_path=path,
                path_length=len(path),
                observations=observations,
                error=None if success else "Did not win within step limit"
            )
                
        except Exception as e:
            import traceback
            return ExplorationResult(
                game_file=game_file,
                success=False,
                optimal_path=[],
                path_length=0,
                observations=[],
                error=f"Exploration error: {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            try:
                env.close()
            except:
                pass
    
    def explore_with_bfs(self, game_file: str, verbose: bool = False) -> ExplorationResult:
        """
        Use BFS to find the shortest path.
        This is slower but guarantees finding the optimal path.
        """
        from collections import deque
        import hashlib
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Exploring with BFS: {game_file}")
            print(f"{'='*60}")
        
        try:
            env = self._create_env(game_file, with_expert=False)
        except Exception as e:
            return ExplorationResult(
                game_file=game_file,
                success=False,
                optimal_path=[],
                path_length=0,
                observations=[],
                error=f"Failed to create env: {str(e)}"
            )
        
        try:
            queue = deque()
            queue.append([])
            
            visited_states = set()
            winning_path = None
            states_explored = 0
            
            while queue and winning_path is None:
                path = queue.popleft()
                
                # Replay path to get current state
                obs, infos = env.reset()
                for action in path:
                    obs, _, done, infos = env.step(action)
                
                # Hash current state
                facts = tuple(sorted(
                    f"{fact.name} " + " ".join(name.strip() for name in fact.names)
                    for fact in infos.get("facts", [])
                ))
                state_hash = hashlib.md5(str(facts).encode()).hexdigest()[:16]
                
                if state_hash in visited_states:
                    continue
                visited_states.add(state_hash)
                states_explored += 1
                
                if verbose and states_explored % 50 == 0:
                    print(f"  States explored: {states_explored}, queue size: {len(queue)}, depth: {len(path)}")
                
                # Check if won
                if infos.get("won", False):
                    winning_path = path
                    if verbose:
                        print(f"  Found winning path with {len(path)} steps!")
                    break
                
                # Limit depth
                if len(path) >= self.max_steps:
                    continue
                
                # Add children to queue
                admissible = infos.get("admissible_commands", [])
                for action in admissible:
                    new_path = path + [action]
                    queue.append(new_path)
            
            env.close()
            
            if winning_path is not None:
                # Replay to get observations
                env = self._create_env(game_file, with_expert=False)
                obs, infos = env.reset()
                observations = [obs]
                for action in winning_path:
                    obs, _, _, infos = env.step(action)
                    observations.append(obs)
                env.close()
                
                return ExplorationResult(
                    game_file=game_file,
                    success=True,
                    optimal_path=winning_path,
                    path_length=len(winning_path),
                    observations=observations
                )
            else:
                return ExplorationResult(
                    game_file=game_file,
                    success=False,
                    optimal_path=[],
                    path_length=0,
                    observations=[],
                    error=f"No winning path found. Explored {states_explored} states."
                )
                
        except Exception as e:
            import traceback
            return ExplorationResult(
                game_file=game_file,
                success=False,
                optimal_path=[],
                path_length=0,
                observations=[],
                error=f"Exploration error: {str(e)}\n{traceback.format_exc()}"
            )
        finally:
            try:
                env.close()
            except:
                pass
    
    def explore_game(self, game_file: str, verbose: bool = False) -> ExplorationResult:
        """Explore a game using the configured method."""
        if self.use_expert:
            return self.explore_with_expert(game_file, verbose)
        else:
            return self.explore_with_bfs(game_file, verbose)


def find_game_files(data_path: str) -> List[str]:
    """Find all game.tw-pddl files in the data path."""
    game_files = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f == "game.tw-pddl":
                game_files.append(os.path.join(root, f))
    return sorted(game_files)


def load_game_files_from_split(data_path: str, split: str) -> List[str]:
    """Load game files from AgentBench's split json file."""
    split_file = os.path.join("data/alfworld", f"{split}.json")
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, "r") as f:
        content = json.load(f)
    
    game_files = []
    for task_type, files in content.items():
        for file in files:
            full_path = os.path.join(data_path, file)
            if os.path.exists(full_path):
                game_files.append(full_path)
            else:
                print(f"Warning: Game file not found: {full_path}")
    
    return game_files


def main():
    parser = argparse.ArgumentParser(description="DFS Explorer for ALFWorld (AgentBench)")
    parser.add_argument("--game_file", type=str, help="Path to a single game.tw-pddl file")
    parser.add_argument("--data_path", type=str, help="Path to ALFWORLD_DATA directory")
    parser.add_argument("--split", type=str, choices=["dev", "standard"], 
                        help="Use AgentBench split file (dev or standard)")
    parser.add_argument("--output", type=str, default="knowledge_base.json", 
                        help="Output file for knowledge base")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum steps per game")
    parser.add_argument("--method", type=str, default="expert", choices=["expert", "bfs"], 
                        help="Exploration method: 'expert' (fast) or 'bfs' (complete)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of games to explore")
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data_path:
        data_path = os.path.expandvars(args.data_path)
    else:
        # Try to get from environment or config
        data_path = os.environ.get("ALFWORLD_DATA")
        if not data_path:
            # Try default location
            config = load_config("src/server/tasks/alfworld/configs/base_config.yaml")
            data_path = os.path.expandvars(config.get("dataset", {}).get("data_path", ""))
        if not data_path:
            print("Error: Please set ALFWORLD_DATA environment variable or use --data_path")
            return
    
    # Set environment variable for alfworld
    os.environ["ALFWORLD_DATA"] = data_path
    
    use_expert = (args.method == "expert")
    explorer = ALFWorldExplorer(max_steps=args.max_steps, use_expert=use_expert)
    
    if args.game_file:
        # Single game mode
        result = explorer.explore_game(args.game_file, verbose=True)
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"  Success: {result.success}")
        print(f"  Path length: {result.path_length}")
        if result.success:
            print(f"  Optimal path:")
            for i, action in enumerate(result.optimal_path):
                print(f"    {i+1}. {action}")
        if result.error:
            print(f"  Error: {result.error}")
            
    elif args.split:
        # Use AgentBench split file
        game_files = load_game_files_from_split(data_path, args.split)
        
        if args.limit:
            game_files = game_files[:args.limit]
        
        print(f"Found {len(game_files)} game files from {args.split} split")
        print(f"Using method: {args.method}")
        print(f"Data path: {data_path}")
        
        knowledge_base = {}
        success_count = 0
        
        for i, game_file in enumerate(game_files):
            print(f"\n[{i+1}/{len(game_files)}] {os.path.basename(os.path.dirname(game_file))}")
            
            result = explorer.explore_game(game_file, verbose=args.verbose)
            
            # Store result with relative path
            relative_path = os.path.relpath(game_file, data_path)
            knowledge_base[relative_path] = {
                "success": result.success,
                "optimal_path": result.optimal_path,
                "path_length": result.path_length,
                "observations": result.observations,
                "error": result.error
            }
            
            if result.success:
                success_count += 1
                print(f"  ✓ Success! Path length: {result.path_length}")
            else:
                print(f"  ✗ Failed: {result.error}")
        
        # Save knowledge base
        with open(args.output, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total games: {len(game_files)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(game_files) - success_count}")
        print(f"  Success rate: {success_count/len(game_files)*100:.1f}%")
        print(f"  Knowledge base saved to: {args.output}")
        
    elif args.data_path:
        # Batch mode with data_path
        game_files = find_game_files(data_path)
        
        if args.limit:
            game_files = game_files[:args.limit]
        
        print(f"Found {len(game_files)} game files")
        print(f"Using method: {args.method}")
        
        knowledge_base = {}
        success_count = 0
        
        for i, game_file in enumerate(game_files):
            print(f"\n[{i+1}/{len(game_files)}] {os.path.basename(os.path.dirname(game_file))}")
            
            result = explorer.explore_game(game_file, verbose=args.verbose)
            
            relative_path = os.path.relpath(game_file, data_path)
            knowledge_base[relative_path] = {
                "success": result.success,
                "optimal_path": result.optimal_path,
                "path_length": result.path_length,
                "observations": result.observations,
                "error": result.error
            }
            
            if result.success:
                success_count += 1
                print(f"  ✓ Success! Path length: {result.path_length}")
            else:
                print(f"  ✗ Failed: {result.error}")
        
        # Save knowledge base
        with open(args.output, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total games: {len(game_files)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(game_files) - success_count}")
        print(f"  Success rate: {success_count/len(game_files)*100:.1f}%")
        print(f"  Knowledge base saved to: {args.output}")
        
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Single game:")
        print("  python -m src.server.tasks.alfworld.dfs_explore --game_file /path/to/game.tw-pddl")
        print()
        print("  # Using AgentBench split:")
        print("  python -m src.server.tasks.alfworld.dfs_explore --split dev --output kb_dev.json")
        print()
        print("  # Batch with data path:")
        print("  python -m src.server.tasks.alfworld.dfs_explore --data_path $ALFWORLD_DATA/json_2.1.1/valid_unseen --output kb.json")


if __name__ == "__main__":
    main()

