"""
SWE-bench Integration Module
Handles loading and parsing SWE-bench data and agent trajectories
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class SWEBenchLoader:
    """
    Loader for SWE-bench dataset and agent trajectories
    """

    def __init__(self, data_dir: str = "data/swebench"):
        """
        Initialize SWE-bench loader

        Args:
            data_dir: Directory containing SWE-bench data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_instance(self, instance_id: str, dataset_file: str = "swebench_lite.json") -> Optional[Dict]:
        """
        Load a specific SWE-bench instance

        Args:
            instance_id: Instance ID (e.g., "django__django-12345")
            dataset_file: Dataset filename

        Returns:
            Instance dictionary or None if not found
        """
        dataset_path = self.data_dir / dataset_file
        if not dataset_path.exists():
            print(f"Dataset file not found: {dataset_path}")
            return None

        with open(dataset_path, 'r', encoding='utf-8') as f:
            instances = json.load(f)

        for instance in instances:
            if instance.get('instance_id') == instance_id:
                return instance

        return None

    def load_trajectory(self, trajectory_file: str) -> Dict[str, Any]:
        """
        Load agent trajectory from file

        Args:
            trajectory_file: Path to trajectory JSON file

        Returns:
            Trajectory dictionary
        """
        traj_path = Path(trajectory_file)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

        with open(traj_path, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)

        return trajectory

    def parse_swe_agent_trajectory(self, trajectory: Dict) -> Dict[str, Any]:
        """
        Parse SWE-agent trajectory format into standardized format

        Expected SWE-agent format:
        {
            "instance_id": "django__django-12345",
            "model_name": "gpt-4",
            "trajectory": [
                {
                    "action": "search_file 'settings.py'",
                    "observation": "Found 3 files: ...",
                    "thought": "I need to find the settings file"
                },
                ...
            ],
            "result": {
                "success": false,
                "error": "Tests failed: 3/5 passing"
            }
        }

        Returns:
            Standardized trajectory format
        """
        instance_id = trajectory.get('instance_id', 'unknown')
        model_name = trajectory.get('model_name', 'unknown')
        raw_trajectory = trajectory.get('trajectory', [])
        result = trajectory.get('result', {})

        # Convert to standardized format
        standardized = {
            'instance_id': instance_id,
            'model_name': model_name,
            'domain': 'code',
            'task_description': trajectory.get('task_description', ''),
            'steps': [],
            'final_result': {
                'success': result.get('success', False),
                'error_message': result.get('error', ''),
                'tests_passed': result.get('tests_passed', 0),
                'tests_total': result.get('tests_total', 0)
            }
        }

        # Parse each step
        for i, step in enumerate(raw_trajectory):
            standardized_step = {
                'step_number': i + 1,
                'observation': step.get('observation', ''),
                'thought': step.get('thought', ''),
                'action': step.get('action', ''),
                'modules': {
                    'memory': step.get('thought', ''),  # Use thought as proxy for memory
                    'reflection': self._extract_reflection(step.get('thought', '')),
                    'planning': self._extract_planning(step.get('thought', '')),
                    'action': step.get('action', '')
                }
            }
            standardized['steps'].append(standardized_step)

        return standardized

    def parse_generic_trajectory(self, trajectory: Dict) -> Dict[str, Any]:
        """
        Parse generic trajectory format

        Flexible format for different code agents
        """
        # Try to extract standard fields
        standardized = {
            'instance_id': trajectory.get('instance_id') or trajectory.get('id', 'unknown'),
            'model_name': trajectory.get('model_name') or trajectory.get('model', 'unknown'),
            'domain': trajectory.get('domain', 'code'),
            'task_description': trajectory.get('task_description') or trajectory.get('problem_statement', ''),
            'steps': [],
            'final_result': trajectory.get('final_result') or trajectory.get('result', {})
        }

        # Handle different step formats
        steps = trajectory.get('steps') or trajectory.get('trajectory', [])
        for i, step in enumerate(steps):
            # Flexible step parsing
            standardized_step = {
                'step_number': step.get('step_number', i + 1),
                'observation': step.get('observation') or step.get('output', ''),
                'modules': {}
            }

            # Check if modules are already nested under 'modules' key
            if 'modules' in step and isinstance(step['modules'], dict):
                # Already in correct format
                standardized_step['modules'] = step['modules']
            else:
                # Try to extract module outputs from top level
                if 'memory' in step:
                    standardized_step['modules']['memory'] = step['memory']
                if 'reflection' in step:
                    standardized_step['modules']['reflection'] = step['reflection']
                if 'planning' in step or 'plan' in step:
                    standardized_step['modules']['planning'] = step.get('planning') or step.get('plan', '')
                if 'action' in step:
                    standardized_step['modules']['action'] = step['action']

            standardized['steps'].append(standardized_step)

        return standardized

    def _extract_reflection(self, thought: str) -> str:
        """Extract reflection component from thought (heuristic)"""
        # Look for reflection keywords
        reflection_keywords = ['however', 'but', 'although', 'mistake', 'wrong', 'should have', 'realized']
        sentences = thought.split('.')
        reflection = []

        for sent in sentences:
            if any(kw in sent.lower() for kw in reflection_keywords):
                reflection.append(sent.strip())

        return '. '.join(reflection) if reflection else ''

    def _extract_planning(self, thought: str) -> str:
        """Extract planning component from thought (heuristic)"""
        # Look for planning keywords
        planning_keywords = ['will', 'should', 'need to', 'going to', 'plan', 'next', 'step']
        sentences = thought.split('.')
        planning = []

        for sent in sentences:
            if any(kw in sent.lower() for kw in planning_keywords):
                planning.append(sent.strip())

        return '. '.join(planning) if planning else thought  # Default to full thought

    def load_multiple_trajectories(self, trajectory_dir: str, max_count: Optional[int] = None, start_index: int = 0) -> List[Dict]:
        """
        Load multiple trajectory files from directory

        Args:
            trajectory_dir: Directory containing trajectory JSON files
            max_count: Maximum number of trajectories to load (None = all)
            start_index: Index to start loading from (for batch processing)

        Returns:
            List of standardized trajectories
        """
        traj_dir = Path(trajectory_dir)
        if not traj_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        trajectories = []
        json_files = sorted(list(traj_dir.glob("*.json")))

        # Apply start_index and max_count for batch slicing
        if max_count:
            json_files = json_files[start_index:start_index + max_count]
        elif start_index > 0:
            json_files = json_files[start_index:]

        for json_file in json_files:
            try:
                trajectory = self.load_trajectory(str(json_file))
                # Try SWE-agent format first, fallback to generic
                if 'trajectory' in trajectory:
                    standardized = self.parse_swe_agent_trajectory(trajectory)
                else:
                    standardized = self.parse_generic_trajectory(trajectory)

                trajectories.append(standardized)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return trajectories

    def extract_compilation_errors(self, step: Dict) -> List[str]:
        """
        Extract compilation/syntax errors from step observation

        Args:
            step: Step dictionary

        Returns:
            List of error messages
        """
        observation = step.get('observation', '')
        errors = []

        # Common error patterns
        error_patterns = [
            'SyntaxError',
            'IndentationError',
            'ImportError',
            'ModuleNotFoundError',
            'NameError',
            'AttributeError',
            'TypeError',
            'ValueError',
        ]

        for pattern in error_patterns:
            if pattern in observation:
                # Extract the error line
                lines = observation.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        # Get error line and surrounding context
                        context_start = max(0, i - 1)
                        context_end = min(len(lines), i + 3)
                        error_context = '\n'.join(lines[context_start:context_end])
                        errors.append(error_context)
                        break

        return errors

    def extract_test_results(self, trajectory: Dict) -> Dict[str, Any]:
        """
        Extract test results from trajectory

        Args:
            trajectory: Trajectory dictionary

        Returns:
            Test results dictionary
        """
        final_result = trajectory.get('final_result', {})
        last_step = trajectory.get('steps', [])[-1] if trajectory.get('steps') else {}
        last_observation = last_step.get('observation', '')

        test_results = {
            'success': final_result.get('success', False),
            'tests_passed': final_result.get('tests_passed', 0),
            'tests_total': final_result.get('tests_total', 0),
            'error_message': final_result.get('error_message', ''),
            'failure_details': []
        }

        # Parse test failures from observation
        if 'FAILED' in last_observation or 'ERROR' in last_observation:
            lines = last_observation.split('\n')
            for line in lines:
                if 'FAILED' in line or 'ERROR' in line:
                    test_results['failure_details'].append(line.strip())

        return test_results


def create_sample_trajectory(instance_id: str = "test-001") -> Dict:
    """
    Create a sample trajectory for testing

    Args:
        instance_id: Instance ID

    Returns:
        Sample trajectory dictionary
    """
    return {
        'instance_id': instance_id,
        'model_name': 'gpt-4',
        'domain': 'code',
        'task_description': 'Fix bug in authentication module where users can login with expired tokens',
        'steps': [
            {
                'step_number': 1,
                'observation': 'Repository structure:\n- src/auth.py\n- src/tokens.py\n- tests/test_auth.py',
                'modules': {
                    'memory': 'No prior history.',
                    'reflection': 'This is the first step.',
                    'planning': 'I need to locate the authentication code and understand how token validation works.',
                    'action': 'search_file "auth.py"'
                }
            },
            {
                'step_number': 2,
                'observation': 'Found file: src/auth.py\nContains: def validate_token(token) - checks if token is valid',
                'modules': {
                    'memory': 'I found auth.py in src/ folder.',
                    'reflection': 'Good, I located the main authentication file.',
                    'planning': 'I should read the validate_token function to see if it checks expiration.',
                    'action': 'read_file "src/auth.py"'
                }
            },
            {
                'step_number': 3,
                'observation': 'File contents:\ndef validate_token(token):\n    if token.signature_valid:\n        return True\n    return False',
                'modules': {
                    'memory': 'The validate_token function only checks signature, not expiration.',
                    'reflection': 'This is the bug! The function doesn\'t check if token.is_expired.',
                    'planning': 'I need to add expiration check: if token.is_expired: return False',
                    'action': 'edit_file "src/auth.py" --add_line "if token.is_expired: return False"'
                }
            },
            {
                'step_number': 4,
                'observation': 'AttributeError: Token object has no attribute \'is_expired\'',
                'modules': {
                    'memory': 'I tried to use token.is_expired but it doesn\'t exist.',
                    'reflection': 'Made an error - I hallucinated the is_expired attribute.',
                    'planning': 'I need to check tokens.py to see the correct attribute name.',
                    'action': 'read_file "src/tokens.py"'
                }
            },
            {
                'step_number': 5,
                'observation': 'File contents:\nclass Token:\n    def __init__(self):\n        self.expires_at = None\n        self.signature_valid = False\n    \n    def is_expired_token(self):\n        return datetime.now() > self.expires_at',
                'modules': {
                    'memory': 'The method is called is_expired_token(), not is_expired.',
                    'reflection': 'I should have read tokens.py first before assuming the API.',
                    'planning': 'Use token.is_expired_token() in validation check.',
                    'action': 'edit_file "src/auth.py" --add_line "if token.is_expired_token(): return False"'
                }
            }
        ],
        'final_result': {
            'success': False,
            'error_message': 'Tests failed: 2/5 passing',
            'tests_passed': 2,
            'tests_total': 5
        }
    }
