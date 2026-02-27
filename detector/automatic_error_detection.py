"""
Automatic Error Detection for Code Agent Trajectories
Implements the 95-pattern regex-based detection engine (Channel A).

Patterns are organized by cognitive module:
  Memory (18), Reflection (15), Planning (22), Action (28), System (12)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DetectedError:
    """An error detected by the automatic (regex) engine"""
    module: str
    error_type: str
    confidence: float
    evidence: str


# ============================================================
# Pattern Library — 95 patterns across 5 modules
# ============================================================

# Each entry: (compiled_regex, module, error_type, confidence)
# Patterns match against step observation and/or action text.

_MEMORY_PATTERNS = [
    # dependency_omission (ImportError family)
    (re.compile(r'ImportError:\s*.+', re.IGNORECASE), 'memory', 'dependency_omission', 1.0),
    (re.compile(r'ModuleNotFoundError:\s*.+', re.IGNORECASE), 'memory', 'dependency_omission', 1.0),
    (re.compile(r'cannot import name\s+'), 'memory', 'dependency_omission', 0.95),
    (re.compile(r'No module named\s+'), 'memory', 'dependency_omission', 0.95),
    (re.compile(r'package\s*.*not found', re.IGNORECASE), 'memory', 'dependency_omission', 0.9),
    (re.compile(r'from\s+\S+\s+import\s+.*ImportError', re.IGNORECASE), 'memory', 'dependency_omission', 0.9),
    (re.compile(r'NameError:\s+name\s+[\'\"]\w+[\'\"] is not defined'), 'memory', 'dependency_omission', 0.85),
    (re.compile(r'ModuleNotFoundError'), 'memory', 'dependency_omission', 0.95),
    (re.compile(r'install\s+.*package', re.IGNORECASE), 'memory', 'dependency_omission', 0.7),
    (re.compile(r'pip install\s+\S+.*failed', re.IGNORECASE), 'memory', 'dependency_omission', 0.8),
    (re.compile(r'missing\s+(required\s+)?dependency', re.IGNORECASE), 'memory', 'dependency_omission', 0.9),
    (re.compile(r'could not\s+(find|import)\s+', re.IGNORECASE), 'memory', 'dependency_omission', 0.8),
    # hallucination
    (re.compile(r'has no (attribute|member)\s+[\'\"]\w+[\'\"].*did you mean', re.IGNORECASE), 'memory', 'hallucination', 0.8),
    (re.compile(r'object has no attribute.*\(did you mean', re.IGNORECASE), 'memory', 'hallucination', 0.8),
    # retrieval_failure
    (re.compile(r'file\s+not\s+found.*searching', re.IGNORECASE), 'memory', 'retrieval_failure', 0.7),
    (re.compile(r'No such file or directory.*previously', re.IGNORECASE), 'memory', 'retrieval_failure', 0.7),
    (re.compile(r'FileNotFoundError', re.IGNORECASE), 'memory', 'retrieval_failure', 0.6),
    (re.compile(r'cannot find.*file.*that was', re.IGNORECASE), 'memory', 'retrieval_failure', 0.7),
]

_REFLECTION_PATTERNS = [
    # error_dismissal
    (re.compile(r'ignoring\s+(the\s+)?error', re.IGNORECASE), 'reflection', 'error_dismissal', 0.9),
    (re.compile(r'error\s+(is\s+)?(not\s+)?(important|relevant|critical)', re.IGNORECASE), 'reflection', 'error_dismissal', 0.8),
    (re.compile(r'can\s+(safely\s+)?ignore\s+(this|that|the)\s+error', re.IGNORECASE), 'reflection', 'error_dismissal', 0.85),
    (re.compile(r'this error (is|seems) (harmless|benign|safe)', re.IGNORECASE), 'reflection', 'error_dismissal', 0.85),
    (re.compile(r'despite the error', re.IGNORECASE), 'reflection', 'error_dismissal', 0.7),
    (re.compile(r"let'?s? (ignore|skip|move past) (this|that|the) error", re.IGNORECASE), 'reflection', 'error_dismissal', 0.85),
    (re.compile(r'error.*not.*blocking', re.IGNORECASE), 'reflection', 'error_dismissal', 0.7),
    # repetition_blindness
    (re.compile(r'already tried.*same', re.IGNORECASE), 'reflection', 'repetition_blindness', 0.85),
    (re.compile(r'let me try.*again', re.IGNORECASE), 'reflection', 'repetition_blindness', 0.6),
    (re.compile(r'trying the same (approach|fix|solution)', re.IGNORECASE), 'reflection', 'repetition_blindness', 0.8),
    (re.compile(r'attempt(ing)? the same', re.IGNORECASE), 'reflection', 'repetition_blindness', 0.75),
    (re.compile(r'same (error|issue|problem|failure)\s+(again|persists|recurring)', re.IGNORECASE), 'reflection', 'repetition_blindness', 0.85),
    # outcome_misinterpretation
    (re.compile(r'test.*pass.*but.*(fail|error)', re.IGNORECASE), 'reflection', 'outcome_misinterpretation', 0.85),
    (re.compile(r'tests? (passed|succeeded).*FAILED', re.IGNORECASE), 'reflection', 'outcome_misinterpretation', 0.9),
    (re.compile(r'all tests? pass', re.IGNORECASE), 'reflection', 'outcome_misinterpretation', 0.5),
]

_PLANNING_PATTERNS = [
    # api_hallucination
    (re.compile(r"AttributeError:\s*.*has no attribute\s+['\"]"), 'planning', 'api_hallucination', 0.95),
    (re.compile(r"NameError:\s*.*is not defined"), 'planning', 'api_hallucination', 0.85),
    (re.compile(r'calling.*non-existent', re.IGNORECASE), 'planning', 'api_hallucination', 0.9),
    (re.compile(r'AttributeError:\s+type object', re.IGNORECASE), 'planning', 'api_hallucination', 0.9),
    (re.compile(r"AttributeError:\s+'(\w+)' object has no attribute"), 'planning', 'api_hallucination', 0.95),
    (re.compile(r'no attribute\s+[\'\"]\w+[\'\"]', re.IGNORECASE), 'planning', 'api_hallucination', 0.9),
    (re.compile(r'undefined method', re.IGNORECASE), 'planning', 'api_hallucination', 0.85),
    (re.compile(r'does not (have|support)\s+(method|attribute|function)', re.IGNORECASE), 'planning', 'api_hallucination', 0.85),
    (re.compile(r'not callable', re.IGNORECASE), 'planning', 'api_hallucination', 0.8),
    (re.compile(r'cannot (call|invoke)\s+', re.IGNORECASE), 'planning', 'api_hallucination', 0.8),
    # scope_violation
    (re.compile(r'modif(y|ying|ied)\s+(files?\s+)?(outside|beyond|unrelated)', re.IGNORECASE), 'planning', 'scope_violation', 0.85),
    (re.compile(r'edit(ing|ed)?\s+(a\s+)?different\s+file', re.IGNORECASE), 'planning', 'scope_violation', 0.8),
    (re.compile(r'outside\s+(the\s+)?(scope|task|issue)', re.IGNORECASE), 'planning', 'scope_violation', 0.8),
    (re.compile(r'unrelated\s+(file|module|change)', re.IGNORECASE), 'planning', 'scope_violation', 0.75),
    (re.compile(r'refactor(ing)?\s+(entire|whole|full)', re.IGNORECASE), 'planning', 'scope_violation', 0.7),
    (re.compile(r'changing\s+too\s+many\s+files', re.IGNORECASE), 'planning', 'scope_violation', 0.75),
    # constraint_ignorance
    (re.compile(r'violat(e|es|ing)\s+(the\s+)?(constraint|convention|standard)', re.IGNORECASE), 'planning', 'constraint_ignorance', 0.85),
    (re.compile(r'does not (conform|comply|adhere)', re.IGNORECASE), 'planning', 'constraint_ignorance', 0.8),
    (re.compile(r'backwards?\s*compat', re.IGNORECASE), 'planning', 'constraint_ignorance', 0.7),
    # impossible_action
    (re.compile(r'impossible\s+(to|action)', re.IGNORECASE), 'planning', 'impossible_action', 0.8),
    (re.compile(r'cannot\s+(modify|change|edit)\s+(a\s+)?(read[\s-]?only|immutable|constant)', re.IGNORECASE), 'planning', 'impossible_action', 0.85),
    (re.compile(r'PermissionError', re.IGNORECASE), 'planning', 'impossible_action', 0.75),
]

_ACTION_PATTERNS = [
    # syntax_error
    (re.compile(r'SyntaxError:\s*'), 'action', 'syntax_error', 1.0),
    (re.compile(r'invalid syntax'), 'action', 'syntax_error', 0.95),
    (re.compile(r'unexpected EOF while parsing'), 'action', 'syntax_error', 0.95),
    (re.compile(r'E999\s+SyntaxError'), 'action', 'syntax_error', 1.0),
    (re.compile(r'SyntaxError'), 'action', 'syntax_error', 0.95),
    (re.compile(r"expected.*':'"), 'action', 'syntax_error', 0.85),
    (re.compile(r'unterminated (string|f-string)', re.IGNORECASE), 'action', 'syntax_error', 0.9),
    (re.compile(r'unmatched\s+[\'\"\(\)\[\]\{\}]', re.IGNORECASE), 'action', 'syntax_error', 0.9),
    (re.compile(r'EOL while scanning string literal'), 'action', 'syntax_error', 0.95),
    (re.compile(r'unexpected character after line continuation'), 'action', 'syntax_error', 0.9),
    (re.compile(r'Missing parenthes(is|es) in call', re.IGNORECASE), 'action', 'syntax_error', 0.9),
    (re.compile(r'perhaps you forgot a comma'), 'action', 'syntax_error', 0.85),
    # indentation_error
    (re.compile(r'IndentationError:\s*'), 'action', 'indentation_error', 1.0),
    (re.compile(r'TabError:\s*'), 'action', 'indentation_error', 1.0),
    (re.compile(r'unexpected indent'), 'action', 'indentation_error', 0.95),
    (re.compile(r'expected an indented block'), 'action', 'indentation_error', 0.95),
    (re.compile(r'unindent does not match'), 'action', 'indentation_error', 0.95),
    (re.compile(r'IndentationError'), 'action', 'indentation_error', 0.95),
    (re.compile(r'TabError'), 'action', 'indentation_error', 0.95),
    (re.compile(r'inconsistent use of tabs and spaces'), 'action', 'indentation_error', 0.95),
    # parameter_error
    (re.compile(r'TypeError:\s*.*argument', re.IGNORECASE), 'action', 'parameter_error', 0.9),
    (re.compile(r'TypeError:\s*.*takes?\s+\d+\s+(positional\s+)?argument', re.IGNORECASE), 'action', 'parameter_error', 0.95),
    (re.compile(r'TypeError:\s*.*missing\s+\d+\s+required', re.IGNORECASE), 'action', 'parameter_error', 0.95),
    (re.compile(r'TypeError:\s*.*unexpected keyword argument', re.IGNORECASE), 'action', 'parameter_error', 0.95),
    (re.compile(r'TypeError:\s*.*got multiple values', re.IGNORECASE), 'action', 'parameter_error', 0.9),
    (re.compile(r'TypeError:\s*.*positional argument', re.IGNORECASE), 'action', 'parameter_error', 0.9),
    (re.compile(r"got an unexpected keyword argument\s+['\"]"), 'action', 'parameter_error', 0.95),
    (re.compile(r'wrong number of arguments', re.IGNORECASE), 'action', 'parameter_error', 0.9),
]

_SYSTEM_PATTERNS = [
    # test_timeout
    (re.compile(r'TimeoutError', re.IGNORECASE), 'system', 'test_timeout', 0.95),
    (re.compile(r'TIMEOUT', re.IGNORECASE), 'system', 'test_timeout', 0.9),
    (re.compile(r'timed?\s*out', re.IGNORECASE), 'system', 'test_timeout', 0.85),
    (re.compile(r'execution\s+timed?\s*out', re.IGNORECASE), 'system', 'test_timeout', 0.9),
    (re.compile(r'test.*exceeded.*time\s*limit', re.IGNORECASE), 'system', 'test_timeout', 0.9),
    (re.compile(r'killed.*timeout', re.IGNORECASE), 'system', 'test_timeout', 0.85),
    # tool_execution_error
    (re.compile(r'subprocess.*error', re.IGNORECASE), 'system', 'tool_execution_error', 0.85),
    (re.compile(r'Command.*failed', re.IGNORECASE), 'system', 'tool_execution_error', 0.8),
    (re.compile(r'CalledProcessError', re.IGNORECASE), 'system', 'tool_execution_error', 0.9),
    (re.compile(r'git\s+.*fatal:', re.IGNORECASE), 'system', 'tool_execution_error', 0.85),
    # environment_error
    (re.compile(r'OSError:\s*\[Errno', re.IGNORECASE), 'system', 'environment_error', 0.8),
    (re.compile(r'disk\s+quota\s+exceeded', re.IGNORECASE), 'system', 'environment_error', 0.9),
]

ALL_PATTERNS = (
    _MEMORY_PATTERNS + _REFLECTION_PATTERNS + _PLANNING_PATTERNS +
    _ACTION_PATTERNS + _SYSTEM_PATTERNS
)


class AutomaticErrorDetector:
    """
    Regex-based error detector for agent trajectories.
    Implements 95 patterns across 5 cognitive modules.
    """

    def __init__(self):
        self.patterns = ALL_PATTERNS
        self._seen_errors = {}  # Track repeated errors for repetition_blindness

    def detect_from_output(self, observation: str, step_number: int) -> List[DetectedError]:
        """
        Detect errors from step observation text using regex patterns.

        Args:
            observation: The observation/output text from the step
            step_number: The step number

        Returns:
            List of DetectedError objects
        """
        if not observation:
            return []

        errors = []
        seen_modules = set()  # Only one error per module per step

        for pattern, module, error_type, confidence in self.patterns:
            if module in seen_modules:
                continue

            match = pattern.search(observation)
            if match:
                # Extract evidence around the match
                start = max(0, match.start() - 100)
                end = min(len(observation), match.end() + 200)
                evidence = observation[start:end]

                errors.append(DetectedError(
                    module=module,
                    error_type=error_type,
                    confidence=confidence,
                    evidence=evidence
                ))
                seen_modules.add(module)

        return errors

    def detect_dependency_issues(self, step_data: Dict, previous_steps: List[Dict]) -> Optional[DetectedError]:
        """
        Detect dependency/import issues from step data.

        Args:
            step_data: Current step data
            previous_steps: List of previous step data dicts

        Returns:
            DetectedError or None
        """
        observation = step_data.get('observation', '')

        # Check for ImportError patterns
        import_patterns = [
            (re.compile(r'ImportError:\s*(.+)'), 1.0),
            (re.compile(r'ModuleNotFoundError:\s*(.+)'), 1.0),
            (re.compile(r'cannot import name\s+[\'\"]\w+[\'\"]'), 0.95),
            (re.compile(r'No module named\s+[\'\"]\S+[\'\"]'), 0.95),
        ]

        for pattern, confidence in import_patterns:
            match = pattern.search(observation)
            if match:
                start = max(0, match.start() - 100)
                end = min(len(observation), match.end() + 200)
                evidence = observation[start:end]

                return DetectedError(
                    module='memory',
                    error_type='dependency_omission',
                    confidence=confidence,
                    evidence=evidence
                )

        return None

    def detect_scope_violation(self, step_data: Dict, task_description: str,
                               instance_id: str = None) -> Optional[DetectedError]:
        """
        Detect scope violations by checking if agent edits files unrelated to the task.

        Args:
            step_data: Current step data
            task_description: The task description
            instance_id: Instance ID (contains repo/issue info)

        Returns:
            DetectedError or None
        """
        action = step_data.get('modules', {}).get('action', '')
        observation = step_data.get('observation', '')

        # Check for edit commands on potentially unrelated files
        edit_patterns = [
            re.compile(r'edit\s+(\S+\.py)', re.IGNORECASE),
            re.compile(r'open\s+(\S+\.py)', re.IGNORECASE),
        ]

        # Extract repo name from instance_id (format: owner__repo-issue)
        expected_files = set()
        if instance_id:
            parts = instance_id.split('__')
            if len(parts) >= 2:
                repo_parts = parts[1].rsplit('-', 1)
                if repo_parts:
                    expected_files.add(repo_parts[0].replace('-', '_'))

        # Check for scope violation indicators in observation
        scope_patterns = [
            (re.compile(r'modif(y|ying|ied)\s+(files?\s+)?(outside|beyond)', re.IGNORECASE), 0.85),
            (re.compile(r'edit(ing|ed)?\s+.*test.*(?!that was)', re.IGNORECASE), 0.6),
        ]

        combined = action + ' ' + observation
        for pattern, confidence in scope_patterns:
            match = pattern.search(combined)
            if match:
                start = max(0, match.start() - 50)
                end = min(len(combined), match.end() + 100)
                evidence = combined[start:end]

                return DetectedError(
                    module='planning',
                    error_type='scope_violation',
                    confidence=confidence,
                    evidence=evidence
                )

        return None

    def detect_reflection_errors(self, step_data: Dict,
                                  previous_steps: List[Dict]) -> Optional[DetectedError]:
        """
        Detect reflection errors such as repetition blindness or error dismissal.

        Args:
            step_data: Current step data
            previous_steps: List of previous step data dicts

        Returns:
            DetectedError or None
        """
        observation = step_data.get('observation', '')
        action = step_data.get('modules', {}).get('action', '')

        # Check for repetition blindness: same error appearing multiple times
        if len(previous_steps) >= 2:
            current_errors = self._extract_error_signatures(observation)

            if current_errors:
                consecutive_repeats = 0
                for prev_step in reversed(previous_steps[-5:]):
                    prev_obs = prev_step.get('observation', '')
                    prev_errors = self._extract_error_signatures(prev_obs)

                    if current_errors & prev_errors:
                        consecutive_repeats += 1
                    else:
                        break

                if consecutive_repeats >= 2:
                    shared = current_errors & self._extract_error_signatures(
                        previous_steps[-1].get('observation', '')
                    )
                    error_name = next(iter(shared)) if shared else "unknown"

                    return DetectedError(
                        module='reflection',
                        error_type='repetition_blindness',
                        confidence=0.85,
                        evidence=f"Same error ({error_name}) repeated {consecutive_repeats + 1} times consecutively (first detection)"
                    )

        # Check for error dismissal patterns
        combined = action + ' ' + observation
        dismissal_patterns = [
            (re.compile(r'ignoring\s+(the\s+)?error', re.IGNORECASE), 0.9),
            (re.compile(r'can\s+(safely\s+)?ignore', re.IGNORECASE), 0.85),
            (re.compile(r'despite the error', re.IGNORECASE), 0.7),
            (re.compile(r"let'?s? (ignore|skip|move past).*error", re.IGNORECASE), 0.85),
        ]

        for pattern, confidence in dismissal_patterns:
            match = pattern.search(combined)
            if match:
                start = max(0, match.start() - 50)
                end = min(len(combined), match.end() + 100)
                evidence = combined[start:end]

                return DetectedError(
                    module='reflection',
                    error_type='error_dismissal',
                    confidence=confidence,
                    evidence=evidence
                )

        return None

    def _extract_error_signatures(self, text: str) -> set:
        """Extract error type signatures from output text for comparison"""
        signatures = set()
        error_patterns = [
            (re.compile(r'(SyntaxError)'), 'SyntaxError'),
            (re.compile(r'(IndentationError)'), 'IndentationError'),
            (re.compile(r'(ImportError)'), 'ImportError'),
            (re.compile(r'(ModuleNotFoundError)'), 'ModuleNotFoundError'),
            (re.compile(r'(AttributeError)'), 'AttributeError'),
            (re.compile(r'(TypeError)'), 'TypeError'),
            (re.compile(r'(NameError)'), 'NameError'),
            (re.compile(r'(ValueError)'), 'ValueError'),
            (re.compile(r'(KeyError)'), 'KeyError'),
            (re.compile(r'(FileNotFoundError)'), 'FileNotFoundError'),
            (re.compile(r'(TimeoutError)'), 'TimeoutError'),
        ]
        for pattern, name in error_patterns:
            if pattern.search(text):
                signatures.add(name)
        return signatures


class HybridErrorDetector:
    """
    Combines automatic (regex) detection with LLM-based detection.
    Uses regex as the fast first pass; defers to LLM only when regex finds nothing.
    """

    def __init__(self, auto_detector: AutomaticErrorDetector = None, llm=None):
        self.auto_detector = auto_detector or AutomaticErrorDetector()
        self.llm = llm

    def detect(self, step_data: Dict, task_description: str,
               previous_steps: List[Dict]) -> List[DetectedError]:
        """
        Detect errors using hybrid approach: regex first, then LLM fallback.

        Args:
            step_data: Current step data
            task_description: Task description
            previous_steps: Previous step data

        Returns:
            List of DetectedError objects
        """
        observation = step_data.get('observation', '')
        step_number = step_data.get('step_number', 0)

        # Try automatic detection first
        auto_errors = self.auto_detector.detect_from_output(observation, step_number)

        if auto_errors:
            return auto_errors

        # If no automatic errors and LLM is available, could defer to LLM
        # (In the dual-channel pipeline, this is handled at a higher level)
        return []
