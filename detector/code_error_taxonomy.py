"""
Code-Specific Error Taxonomy Extension
Extends AgentErrorTaxonomy with 12 new error types for code repair domain
Based on SWE-bench analysis
"""

from typing import Dict, List, Any

# Code-specific error definitions extending the original taxonomy
CODE_ERROR_DEFINITIONS = {
    'memory': {
        'file_location_forgetting': {
            'description': 'Agent forgets which file contains specific code or where it made previous edits',
            'example': 'Agent searches for a function definition in wrong file after previously locating it',
            'detection_hints': ['revisits already-explored files', 'searches in wrong directories', 'loses track of file structure']
        },
        'dependency_omission': {
            'description': 'Agent forgets about required imports, dependencies, or prerequisites',
            'example': 'Agent uses a function but forgets to import the required module',
            'detection_hints': ['ImportError', 'NameError', 'missing import statements', 'undefined names']
        },
        'over_simplification': {
            'description': 'Important code context or requirements lost during memory compression',
            'example': 'Agent forgets edge cases or special conditions mentioned in issue description',
            'detection_hints': ['incomplete fix', 'missing edge case handling', 'ignores constraints']
        },
        'hallucination': {
            'description': 'False information about code state, functions, or variables',
            'example': 'Agent remembers a function signature that doesn\'t exist',
            'detection_hints': ['references non-existent code', 'incorrect assumptions about implementation']
        },
        'retrieval_failure': {
            'description': 'Failed to recall relevant code context from previous exploration',
            'example': 'Agent doesn\'t recall that a similar bug was fixed in another file',
            'detection_hints': ['redundant file reads', 'repeats searches', 'misses patterns']
        }
    },
    'reflection': {
        'progress_misjudge': {
            'description': 'Incorrect assessment of fix completion or test status',
            'example': 'Agent thinks all tests pass when some are still failing',
            'detection_hints': ['declares success prematurely', 'ignores failing tests', 'overestimates completion']
        },
        'outcome_misinterpretation': {
            'description': 'Misunderstanding compilation errors or test failures',
            'example': 'Agent misinterprets SyntaxError message and fixes wrong line',
            'detection_hints': ['wrong error interpretation', 'fixes unrelated code', 'misreads stack traces']
        }
    },
    'planning': {
        'scope_violation': {
            'description': 'Plan modifies files or code outside the issue scope',
            'example': 'Issue is about fixing auth.py but agent plans to refactor entire authentication system',
            'detection_hints': ['modifies too many files', 'changes unrelated code', 'overengineering']
        },
        'api_hallucination': {
            'description': 'Plans to use non-existent functions, methods, or APIs',
            'example': 'Agent plans to call utils.parse_json() but function doesn\'t exist in codebase',
            'detection_hints': ['uses undefined functions', 'invents method names', 'AttributeError after execution']
        },
        'constraint_ignorance': {
            'description': 'Plan violates coding standards, backward compatibility, or project conventions',
            'example': 'Plan ignores requirement to maintain Python 2.7 compatibility',
            'detection_hints': ['breaks conventions', 'ignores style guide', 'violates requirements']
        },
        'impossible_action': {
            'description': 'Plans technically impossible code changes',
            'example': 'Plans to modify a read-only constant or built-in function',
            'detection_hints': ['attempts to modify immutables', 'violates language constraints']
        },
        'inefficient_plan': {
            'description': 'Overly complex solution when simpler fix exists',
            'example': 'Plans complete refactor when one-line fix would work',
            'detection_hints': ['overcomplicates', 'unnecessary abstraction', 'excessive changes']
        },
        'redundant_plan': {
            'description': 'Plans to implement already-completed changes',
            'example': 'Plans to add error handling that was already added in previous step',
            'detection_hints': ['repeats fixes', 'redundant modifications', 'ignores previous changes']
        }
    },
    'action': {
        'syntax_error': {
            'description': 'Generated code has syntax errors',
            'example': 'Missing parenthesis, incorrect indentation, invalid Python syntax',
            'detection_hints': ['SyntaxError', 'IndentationError', 'TabError', 'compilation fails']
        },
        'indentation_error': {
            'description': 'Incorrect indentation in Python code (common LLM mistake)',
            'example': 'Mixed tabs and spaces, wrong indent level for nested blocks',
            'detection_hints': ['IndentationError', 'TabError', 'unexpected indent']
        },
        'wrong_file_edit': {
            'description': 'Edits wrong file or wrong location within file',
            'example': 'Meant to edit src/utils.py but edits tests/test_utils.py instead',
            'detection_hints': ['edit location mismatch', 'wrong file modified', 'target not found']
        },
        'format_error': {
            'description': 'Action format doesn\'t match required structure',
            'example': 'Returns code snippet instead of proper edit command format',
            'detection_hints': ['malformed commands', 'invalid action format', 'parse errors']
        },
        'parameter_error': {
            'description': 'Invalid parameters in edit actions',
            'example': 'Uses non-existent line numbers or invalid string replacement',
            'detection_hints': ['line number out of range', 'string not found', 'invalid regex']
        },
        'misalignment': {
            'description': 'Action doesn\'t match the plan',
            'example': 'Plan says fix bug in function X, but action modifies function Y',
            'detection_hints': ['action-plan mismatch', 'implements wrong change']
        }
    },
    'system': {
        'compilation_timeout': {
            'description': 'Code compilation takes too long and times out',
            'example': 'Generated code causes infinite loop during compilation or import',
            'detection_hints': ['timeout during compilation', 'hanging process', 'resource limit']
        },
        'test_timeout': {
            'description': 'Test execution exceeds time limit',
            'example': 'Generated code causes tests to hang or run infinitely',
            'detection_hints': ['test timeout', 'hanging tests', 'infinite loop in tests']
        },
        'step_limit_exhaustion': {
            'description': 'Reached maximum step limit before completing fix',
            'example': 'Agent hits 50-step cap while debugging complex issue',
            'detection_hints': ['max steps reached', 'incomplete fix', 'ran out of actions']
        },
        'tool_execution_error': {
            'description': 'External tool (git, pytest, linter) failed',
            'example': 'pytest crashes or git command fails',
            'detection_hints': ['tool crash', 'command error', 'external failure']
        },
        'environment_error': {
            'description': 'Development environment behaved unexpectedly',
            'example': 'Docker container crash or dependency installation failure',
            'detection_hints': ['environment inconsistency', 'setup failure', 'system error']
        }
    }
}


class CodeErrorTaxonomy:
    """
    Code-specific error taxonomy for SWE-bench domain
    Extends original AgentErrorTaxonomy with 12 new error types
    """

    def __init__(self):
        self.taxonomy = CODE_ERROR_DEFINITIONS

    def get_all_modules(self) -> List[str]:
        """Get list of all modules"""
        return list(self.taxonomy.keys())

    def get_module_errors(self, module: str) -> List[str]:
        """Get all error types for a module"""
        return list(self.taxonomy.get(module, {}).keys())

    def get_error_definition(self, module: str, error_type: str) -> Dict[str, Any]:
        """Get definition of specific error"""
        return self.taxonomy.get(module, {}).get(error_type, {})

    def get_all_errors(self) -> List[tuple]:
        """Get flat list of all (module, error_type) pairs"""
        all_errors = []
        for module, errors in self.taxonomy.items():
            for error_type in errors.keys():
                all_errors.append((module, error_type))
        return all_errors

    def get_detection_hints(self, module: str, error_type: str) -> List[str]:
        """Get detection hints for automatic error identification"""
        error_def = self.get_error_definition(module, error_type)
        return error_def.get('detection_hints', [])

    def format_for_prompt(self, module: str = None) -> str:
        """Format taxonomy for LLM prompts"""
        if module:
            errors = self.taxonomy.get(module, {})
            lines = [f"**{module.upper()} Module Errors (Code Domain):**"]
            for error_type, defn in errors.items():
                lines.append(f"- {error_type}: {defn['description']}")
                lines.append(f"  Example: {defn['example']}")
            return "\n".join(lines)
        else:
            lines = []
            for mod, errors in self.taxonomy.items():
                lines.append(f"\n**{mod.upper()} Module (Code Domain):**")
                for error_type, defn in errors.items():
                    lines.append(f"- {error_type}: {defn['description']}")
            return "\n".join(lines)

    def format_for_phase1_prompt(self, module_name: str) -> str:
        """Format error definitions for Phase 1 detection prompt"""
        errors = self.taxonomy.get(module_name, {})
        if not errors:
            return ""

        lines = [f"# {module_name.upper()} Module Errors:\n"]
        for error_type, defn in errors.items():
            lines.append(f"## {error_type}")
            lines.append(f"**Description:** {defn['description']}")
            lines.append(f"**Example:** {defn['example']}")
            if 'detection_hints' in defn:
                lines.append(f"**Detection Hints:** {', '.join(defn['detection_hints'])}")
            lines.append("")

        return "\n".join(lines)

    def get_code_specific_errors(self) -> List[tuple]:
        """
        Get only the NEW code-specific error types (not in original taxonomy)

        Returns:
            List of (module, error_type) tuples for code-specific errors
        """
        code_specific = [
            ('memory', 'file_location_forgetting'),
            ('memory', 'dependency_omission'),
            ('planning', 'scope_violation'),
            ('planning', 'api_hallucination'),
            ('action', 'syntax_error'),
            ('action', 'indentation_error'),
            ('action', 'wrong_file_edit'),
            ('system', 'compilation_timeout'),
            ('system', 'test_timeout'),
        ]
        return code_specific

    def is_automatically_detectable(self, module: str, error_type: str) -> bool:
        """
        Check if error can be automatically detected from compiler/test output

        Returns:
            True if error can be detected without LLM
        """
        auto_detectable = {
            ('memory', 'dependency_omission'),  # ImportError, NameError
            ('action', 'syntax_error'),  # SyntaxError
            ('action', 'indentation_error'),  # IndentationError
            ('action', 'parameter_error'),  # Runtime errors with clear messages
            ('system', 'compilation_timeout'),  # Timeout signals
            ('system', 'test_timeout'),  # Test timeout signals
            ('system', 'tool_execution_error'),  # Tool error codes
        }
        return (module, error_type) in auto_detectable


# Mapping from compiler/test output patterns to error types
AUTOMATIC_DETECTION_PATTERNS = {
    'SyntaxError': ('action', 'syntax_error'),
    'IndentationError': ('action', 'indentation_error'),
    'TabError': ('action', 'indentation_error'),
    'ImportError': ('memory', 'dependency_omission'),
    'ModuleNotFoundError': ('memory', 'dependency_omission'),
    'NameError': ('memory', 'dependency_omission'),
    'AttributeError': ('planning', 'api_hallucination'),
    'TimeoutError': ('system', 'test_timeout'),
    'TIMEOUT': ('system', 'test_timeout'),
    'Compilation timeout': ('system', 'compilation_timeout'),
}


def detect_error_from_output(output: str) -> tuple:
    """
    Automatically detect error type from compiler/test output

    Args:
        output: Compiler or test output string

    Returns:
        (module, error_type) tuple or None if no match
    """
    for pattern, error_tuple in AUTOMATIC_DETECTION_PATTERNS.items():
        if pattern in output:
            return error_tuple
    return None
