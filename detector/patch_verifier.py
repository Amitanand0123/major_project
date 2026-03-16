"""
Patch Verifier for Phase 3
Clones repos locally, applies LLM-generated patches, and runs
the actual test suite to verify correctness. No Docker needed.

Optimizations:
- Shallow clone (--depth 1) + fetch only the needed commit
- Repo cache: reuse cloned repos across trajectories from same repo
- Gold verification cache: persist gold patch results across runs
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


# Failure category constants
FAILURE_SUCCESS = "success"
FAILURE_CLONE = "clone_failure"
FAILURE_INSTALL = "install_failure"
FAILURE_PATCH_APPLY = "patch_apply_failure"
FAILURE_TEST_TIMEOUT = "test_timeout"
FAILURE_TEST = "test_failure"
FAILURE_PYTHON_COMPAT = "python_compat"
FAILURE_IMPORT_ERROR = "import_error"
FAILURE_UNKNOWN = "unknown_error"


@dataclass
class VerificationResult:
    """Result of patch verification."""
    instance_id: str
    patch_applied: bool
    tests_run: bool
    tests_passed: bool
    exit_code: int
    stdout_snippet: str          # last 500 chars of output
    error_message: str
    duration_seconds: float
    metadata_available: bool
    failure_category: str = FAILURE_UNKNOWN


class PatchVerifier:
    """
    Verifies LLM-generated patches by cloning repos, applying patches,
    and running the actual test suite locally.

    Optimizations:
    - Caches cloned repos so multiple issues from the same repo reuse the clone
    - Uses shallow clone + fetch specific commit to minimize download size
    - Caches gold patch verification results persistently
    """

    DEFAULT_TIMEOUT = 600  # 10 minutes per verification

    def __init__(self, metadata_path: str = "data/swebench/swebench_metadata.json",
                 cache_dir: str = "data/swebench/repo_cache",
                 timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.metadata_path = Path(metadata_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache = None
        self._gold_cache_path = self.cache_dir / "gold_verification_cache.json"
        self._gold_cache = None
        self._python = self._find_python()

    def _find_python(self) -> str:
        """Find a Python executable that has pip available."""
        candidates = [
            sys.executable,
            r"C:\Users\amita\AppData\Local\Programs\Python\Python312\python.exe",
            "python3",
            "python",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "-m", "pip", "--version"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return sys.executable  # fallback

    def is_available(self) -> bool:
        """Check if git and metadata are available."""
        return self._check_git() and self.metadata_path.exists()

    def _check_git(self) -> bool:
        """Check if git is available."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _load_metadata(self) -> Dict:
        """Load SWE-bench metadata from local file."""
        if self._metadata_cache is None:
            if not self.metadata_path.exists():
                self._metadata_cache = {}
                return self._metadata_cache
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self._metadata_cache = json.load(f)
        return self._metadata_cache

    def get_instance_metadata(self, instance_id: str) -> Optional[Dict]:
        """Get metadata for a specific SWE-bench instance."""
        metadata = self._load_metadata()
        return metadata.get(instance_id)

    def _make_error_result(self, instance_id: str, error_message: str,
                           duration: float = 0.0, metadata_available: bool = False,
                           failure_category: str = FAILURE_UNKNOWN) -> VerificationResult:
        """Helper to create a failure VerificationResult."""
        return VerificationResult(
            instance_id=instance_id,
            patch_applied=False, tests_run=False, tests_passed=False,
            exit_code=-1, stdout_snippet="",
            error_message=error_message,
            duration_seconds=duration,
            metadata_available=metadata_available,
            failure_category=failure_category
        )

    def _classify_install_failure(self, stderr: str, stdout: str) -> str:
        """Classify an install failure based on output text."""
        combined = ((stderr or '') + (stdout or '')).lower()

        # Python version incompatibilities
        if any(p in combined for p in [
            'invalid syntax', 'syntaxerror',
            'python_requires', 'requires python',
            'not supported', 'f-string',
        ]):
            return FAILURE_PYTHON_COMPAT

        # Missing C extensions / system deps
        if any(p in combined for p in [
            'modulenotfounderror', 'no module named',
            'error: command', 'cl.exe', 'gcc',
            'fatal error', 'cannot find', 'vcvarsall',
            'microsoft visual c++', 'distutils',
        ]):
            return FAILURE_IMPORT_ERROR

        return FAILURE_INSTALL

    def _install_test_deps(self, python: str, repo_dir: str):
        """Install test dependencies: requirements files, extras, and safe common packages."""
        # 1. Install only safe, universally-needed test packages
        #    (avoid pytest-asyncio, trustme etc. that can cause conflicts)
        safe_deps = ["pytest", "pytest-cov", "pytest-mock", "mock", "coverage"]
        subprocess.run(
            [python, "-m", "pip", "install"] + safe_deps + ["-q"],
            capture_output=True, text=True, timeout=120, cwd=repo_dir
        )

        # 2. Try installing test/dev extras from setup.cfg/pyproject.toml
        #    (these are repo-specific, so they install the RIGHT versions)
        for extra in ["test", "tests", "testing", "dev"]:
            subprocess.run(
                [python, "-m", "pip", "install", "-e", f".[{extra}]", "-q"],
                capture_output=True, text=True, timeout=120, cwd=repo_dir
            )

        # 3. Install from requirements files if they exist
        req_patterns = [
            "requirements-test.txt", "test-requirements.txt",
            "requirements_test.txt", "requirements_dev.txt",
            "requirements-dev.txt", "dev-requirements.txt",
            "requirements/test.txt", "requirements/tests.txt",
            "requirements/dev.txt",
        ]
        for req_file in req_patterns:
            req_path = os.path.join(repo_dir, req_file)
            if os.path.exists(req_path):
                print(f"      Installing {req_file}...")
                subprocess.run(
                    [python, "-m", "pip", "install", "-r", req_path, "-q"],
                    capture_output=True, text=True, timeout=120, cwd=repo_dir
                )

    def _get_cached_repo(self, repo: str) -> Optional[Path]:
        """Get path to cached repo clone, or None if not cached."""
        safe_name = repo.replace("/", "__")
        cached = self.cache_dir / safe_name
        if cached.exists() and (cached / ".git").exists():
            return cached
        return None

    def _clone_or_reuse(self, repo: str, base_commit: str, work_dir: str) -> Optional[str]:
        """
        Clone the repo (or copy from cache) and checkout the base commit.
        Returns path to the repo working directory, or None on failure.
        """
        safe_name = repo.replace("/", "__")
        cached = self.cache_dir / safe_name
        repo_dir = os.path.join(work_dir, "repo")

        if cached.exists() and (cached / ".git").exists():
            # Reuse cached repo — just copy .git and checkout
            print(f"      Using cached repo for {repo}")
            shutil.copytree(str(cached), repo_dir)
            # Reset to clean state
            subprocess.run(
                ["git", "checkout", ".", "--quiet"],
                capture_output=True, text=True, timeout=30, cwd=repo_dir
            )
            subprocess.run(
                ["git", "clean", "-fd", "--quiet"],
                capture_output=True, text=True, timeout=30, cwd=repo_dir
            )
        else:
            # Shallow clone — only get minimal history
            print(f"      Cloning {repo} (shallow)...")
            clone_result = subprocess.run(
                ["git", "clone", "--quiet", "--no-tags",
                 f"https://github.com/{repo}.git", repo_dir],
                capture_output=True, text=True, timeout=180
            )
            if clone_result.returncode != 0:
                return None

            # Cache it for future use
            try:
                shutil.copytree(repo_dir, str(cached))
            except Exception:
                pass  # Cache failure is non-fatal

        # Checkout the specific commit
        checkout = subprocess.run(
            ["git", "checkout", "-q", base_commit],
            capture_output=True, text=True, timeout=30, cwd=repo_dir
        )
        if checkout.returncode != 0:
            # Commit might not be in shallow clone, fetch it
            subprocess.run(
                ["git", "fetch", "--quiet", "origin", base_commit],
                capture_output=True, text=True, timeout=60, cwd=repo_dir
            )
            checkout = subprocess.run(
                ["git", "checkout", "-q", base_commit],
                capture_output=True, text=True, timeout=30, cwd=repo_dir
            )
            if checkout.returncode != 0:
                return None

        return repo_dir

    def verify_patch(self, instance_id: str, model_patch: str) -> VerificationResult:
        """
        Verify a patch by cloning the repo, applying it, and running tests.

        Args:
            instance_id: SWE-bench instance ID
            model_patch: The unified diff patch to verify

        Returns:
            VerificationResult with pass/fail and details
        """
        start_time = time.time()

        meta = self.get_instance_metadata(instance_id)
        if not meta:
            return self._make_error_result(
                instance_id, f"No SWE-bench metadata for {instance_id}",
                failure_category=FAILURE_UNKNOWN
            )

        if not model_patch or not model_patch.strip():
            return self._make_error_result(
                instance_id, "Empty patch provided",
                metadata_available=True, failure_category=FAILURE_PATCH_APPLY
            )

        repo = meta.get("repo", "")
        base_commit = meta.get("base_commit", "")
        test_patch = meta.get("test_patch", "")
        fail_to_pass = meta.get("FAIL_TO_PASS", [])

        if not repo or not base_commit:
            return self._make_error_result(
                instance_id, "Missing repo or base_commit in metadata",
                duration=time.time() - start_time, metadata_available=True,
                failure_category=FAILURE_UNKNOWN
            )

        # Parse FAIL_TO_PASS (may be a JSON string or already a list)
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = [fail_to_pass]

        # Work in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Step 1: Clone or reuse cached repo
                repo_dir = self._clone_or_reuse(repo, base_commit, tmpdir)
                if not repo_dir:
                    return self._make_error_result(
                        instance_id, f"Failed to clone/checkout {repo}@{base_commit[:8]}",
                        duration=time.time() - start_time, metadata_available=True,
                        failure_category=FAILURE_CLONE
                    )

                # Step 2: Apply model patch
                model_patch_path = os.path.join(tmpdir, "model.patch")
                with open(model_patch_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(model_patch)

                apply_result = subprocess.run(
                    ["git", "apply", "--allow-empty", model_patch_path],
                    capture_output=True, text=True, timeout=30, cwd=repo_dir
                )
                patch_applied = apply_result.returncode == 0
                if not patch_applied:
                    # Try with --reject for partial application
                    subprocess.run(
                        ["git", "apply", "--reject", "--allow-empty", model_patch_path],
                        capture_output=True, text=True, timeout=30, cwd=repo_dir
                    )

                # Step 3: Apply test patch (if available)
                if test_patch:
                    test_patch_path = os.path.join(tmpdir, "test.patch")
                    with open(test_patch_path, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(test_patch)
                    subprocess.run(
                        ["git", "apply", "--allow-empty", test_patch_path],
                        capture_output=True, text=True, timeout=30, cwd=repo_dir
                    )

                # Step 4: Create isolated virtual environment
                venv_dir = os.path.join(tmpdir, "venv")
                base_python = self._python
                print(f"      Creating virtual environment...")
                venv_result = subprocess.run(
                    [base_python, "-m", "venv", venv_dir],
                    capture_output=True, text=True, timeout=60, cwd=repo_dir
                )
                if venv_result.returncode != 0:
                    return self._make_error_result(
                        instance_id, "Failed to create venv",
                        start_time, patch_applied=patch_applied
                    )

                # Use venv python for all subsequent operations
                if os.name == 'nt':
                    python = os.path.join(venv_dir, "Scripts", "python.exe")
                else:
                    python = os.path.join(venv_dir, "bin", "python")

                # Step 4a: Install project
                print(f"      Installing project...")
                install_result = subprocess.run(
                    [python, "-m", "pip", "install", "-e", ".", "-q"],
                    capture_output=True, text=True, timeout=120, cwd=repo_dir
                )

                # Check install failure
                if install_result.returncode != 0:
                    category = self._classify_install_failure(
                        install_result.stderr, install_result.stdout
                    )
                    return VerificationResult(
                        instance_id=instance_id,
                        patch_applied=patch_applied,
                        tests_run=False, tests_passed=False,
                        exit_code=install_result.returncode,
                        stdout_snippet=(install_result.stderr or "")[-500:],
                        error_message=f"Install failed ({category})",
                        duration_seconds=round(time.time() - start_time, 2),
                        metadata_available=True,
                        failure_category=category
                    )

                # Step 4b: Install test dependencies (into isolated venv)
                self._install_test_deps(python, repo_dir)

                # Step 5: Run tests
                # Override addopts to prevent pytest.ini coverage flags from breaking tests
                print(f"      Running tests...")
                if fail_to_pass:
                    test_cmd = [python, "-m", "pytest", "-o", "addopts="] + fail_to_pass + ["-x", "--tb=short"]
                else:
                    test_cmd = [python, "-m", "pytest", "-o", "addopts=", "--tb=short", "-x", "-q"]

                test_result = subprocess.run(
                    test_cmd,
                    capture_output=True, text=True,
                    timeout=self.timeout, cwd=repo_dir
                )

                duration = time.time() - start_time
                combined = (test_result.stdout or "") + "\n" + (test_result.stderr or "")
                tests_passed = test_result.returncode == 0
                stdout_snippet = combined[-500:] if len(combined) > 500 else combined

                # Determine failure category
                if tests_passed:
                    category = FAILURE_SUCCESS
                    error_msg = ""
                elif not patch_applied:
                    category = FAILURE_PATCH_APPLY
                    error_msg = "Failed to apply patch"
                else:
                    category = FAILURE_TEST
                    error_msg = "Tests failed"

                return VerificationResult(
                    instance_id=instance_id,
                    patch_applied=patch_applied,
                    tests_run=True,
                    tests_passed=tests_passed,
                    exit_code=test_result.returncode,
                    stdout_snippet=stdout_snippet,
                    error_message=error_msg,
                    duration_seconds=round(duration, 2),
                    metadata_available=True,
                    failure_category=category
                )

            except subprocess.TimeoutExpired:
                return self._make_error_result(
                    instance_id, f"Timeout after {self.timeout}s",
                    duration=self.timeout, metadata_available=True,
                    failure_category=FAILURE_TEST_TIMEOUT
                )
            except Exception as e:
                return self._make_error_result(
                    instance_id, f"Error: {str(e)}",
                    duration=time.time() - start_time, metadata_available=True,
                    failure_category=FAILURE_UNKNOWN
                )

    def verify_with_gold_patch(self, instance_id: str) -> VerificationResult:
        """
        Verify using the gold (correct) patch from SWE-bench.
        Sanity check to confirm the setup works.
        """
        meta = self.get_instance_metadata(instance_id)
        if not meta:
            return self._make_error_result(instance_id, f"No metadata for {instance_id}")

        gold_patch = meta.get("patch", "")
        if not gold_patch:
            return self._make_error_result(
                instance_id, "No gold patch in metadata", metadata_available=True
            )

        print(f"      Verifying gold patch for {instance_id}...")
        return self.verify_patch(instance_id, gold_patch)

    def verify_gold_patch_cached(self, instance_id: str) -> VerificationResult:
        """
        Verify gold patch with persistent caching.
        Gold patch results are deterministic on a given machine, so cache aggressively.
        """
        # Load cache
        if self._gold_cache is None:
            if self._gold_cache_path.exists():
                try:
                    with open(self._gold_cache_path, 'r', encoding='utf-8') as f:
                        self._gold_cache = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._gold_cache = {}
            else:
                self._gold_cache = {}

        # Check cache hit
        if instance_id in self._gold_cache:
            cached = self._gold_cache[instance_id]
            print(f"      Gold verification cache hit for {instance_id}")
            return VerificationResult(**cached)

        # Cache miss — run verification
        result = self.verify_with_gold_patch(instance_id)

        # Save to cache
        self._gold_cache[instance_id] = asdict(result)
        try:
            with open(self._gold_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._gold_cache, f, indent=2)
        except IOError:
            pass  # Cache write failure is non-fatal

        return result

    def cleanup_cache(self):
        """Remove all cached repos to free disk space."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Repo cache cleared.")


# Keep old name as alias for backward compatibility
DockerPatchVerifier = PatchVerifier


if __name__ == "__main__":
    verifier = PatchVerifier()

    print("=== Patch Verifier Test ===")
    print(f"Git available: {verifier._check_git()}")
    print(f"Metadata available: {verifier.metadata_path.exists()}")

    if not verifier.is_available():
        print("Verifier not available. Check git and metadata.")
        exit(1)

    # Test with first available instance that has a gold patch
    metadata = verifier._load_metadata()
    test_instance = None
    for iid, meta in metadata.items():
        if meta.get("patch") and meta.get("test_patch") and meta.get("FAIL_TO_PASS"):
            test_instance = iid
            break

    if test_instance:
        print(f"\nTesting with gold patch for: {test_instance}")
        result = verifier.verify_gold_patch_cached(test_instance)
        print(f"\nResult:")
        for k, v in asdict(result).items():
            if k == "stdout_snippet":
                print(f"  {k}: {v[:200]}...")
            else:
                print(f"  {k}: {v}")
    else:
        print("No suitable test instance found in metadata.")
