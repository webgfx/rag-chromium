"""
Chromium repository data extraction utilities.
Extracts commits, diffs, file changes, and metadata from the Chromium repository.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from git import Repo, InvalidGitRepositoryError
import pandas as pd
from tqdm import tqdm

from ..core.config import get_config
from ..core.logger import setup_logger, PerformanceLogger


@dataclass
class CommitData:
    """Data structure for a Chromium commit."""
    sha: str
    message: str
    author_name: str
    author_email: str
    commit_date: datetime
    files_changed: List[str]
    additions: int
    deletions: int
    diff: str
    parents: List[str]
    tags: List[str]
    branches: List[str]


@dataclass
class FileChange:
    """Data structure for file changes in a commit."""
    file_path: str
    change_type: str  # A (added), M (modified), D (deleted), R (renamed)
    old_path: Optional[str]
    diff: str
    additions: int
    deletions: int
    language: Optional[str]


class ChromiumDataExtractor:
    """
    Advanced data extractor for Chromium repository.
    Efficiently processes large-scale git history with filtering and optimization.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize the Chromium data extractor.
        
        Args:
            repo_path: Path to the Chromium repository
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.ChromiumDataExtractor")
        
        self.repo_path = Path(repo_path or self.config.data.chromium_repo_path)
        self.cache_dir = Path(self.config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._repo = None
        self._validate_repository()
    
    def _validate_repository(self) -> None:
        """Validate that the repository path is valid."""
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        
        try:
            self._repo = Repo(str(self.repo_path))
            if self._repo.bare:
                raise InvalidGitRepositoryError("Repository is bare")
        except InvalidGitRepositoryError as e:
            raise InvalidGitRepositoryError(f"Invalid git repository at {self.repo_path}: {e}")
        
        self.logger.info(f"Initialized Chromium data extractor for repository: {self.repo_path}")
    
    @property
    def repo(self) -> Repo:
        """Get the Git repository object."""
        if self._repo is None:
            self._repo = Repo(str(self.repo_path))
        return self._repo
    
    def extract_commits(
        self,
        max_count: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        paths: Optional[List[str]] = None,
        branches: Optional[List[str]] = None,
        include_diffs: bool = True,
        batch_size: int = 1000
    ) -> Iterator[List[CommitData]]:
        """
        Extract commits from the repository in batches.
        
        Args:
            max_count: Maximum number of commits to extract
            since: Start date for commit extraction
            until: End date for commit extraction
            paths: Specific file paths to filter commits
            branches: Specific branches to extract from
            include_diffs: Whether to include diff data
            batch_size: Number of commits to yield per batch
        
        Yields:
            Batches of CommitData objects
        """
        with PerformanceLogger(self.logger, "commit extraction"):
            # Build git log arguments
            kwargs = {}
            if max_count:
                kwargs['max_count'] = max_count
            if since:
                kwargs['since'] = since
            if until:
                kwargs['until'] = until
            if paths:
                kwargs['paths'] = paths
            
            # Get commits from specified branches or all branches
            if branches:
                commit_iter = []
                for branch in branches:
                    try:
                        branch_commits = list(self.repo.iter_commits(branch, **kwargs))
                        commit_iter.extend(branch_commits)
                    except Exception as e:
                        self.logger.warning(f"Failed to get commits from branch {branch}: {e}")
                # Remove duplicates based on SHA
                seen_shas = set()
                commit_iter = [c for c in commit_iter if c.hexsha not in seen_shas and not seen_shas.add(c.hexsha)]
            else:
                commit_iter = self.repo.iter_commits(**kwargs)
            
            batch = []
            total_processed = 0
            
            for commit in tqdm(commit_iter, desc="Extracting commits"):
                try:
                    commit_data = self._process_commit(commit, include_diffs)
                    batch.append(commit_data)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                        total_processed += batch_size
                        
                except Exception as e:
                    self.logger.error(f"Failed to process commit {commit.hexsha}: {e}")
                    continue
            
            # Yield remaining commits
            if batch:
                yield batch
                total_processed += len(batch)
            
            self.logger.info(f"Successfully extracted {total_processed} commits")
    
    def _process_commit(self, commit, include_diffs: bool = True) -> CommitData:
        """Process a single commit and extract relevant data."""
        # Get file changes
        files_changed = []
        additions = 0
        deletions = 0
        diff_text = ""
        
        if include_diffs and commit.parents:
            try:
                # Get diff with parent (or first parent for merge commits)
                parent = commit.parents[0]
                diff_index = parent.diff(commit, create_patch=True)
                
                for diff_item in diff_index:
                    if diff_item.a_path:
                        files_changed.append(diff_item.a_path)
                    if diff_item.b_path and diff_item.b_path not in files_changed:
                        files_changed.append(diff_item.b_path)
                    
                    # Get additions/deletions statistics
                    if hasattr(diff_item, 'diff'):
                        diff_content = diff_item.diff.decode('utf-8', errors='ignore')
                        diff_text += f"--- {diff_item.a_path or '/dev/null'}\\n"
                        diff_text += f"+++ {diff_item.b_path or '/dev/null'}\\n"
                        diff_text += diff_content + "\\n\\n"
                        
                        # Count additions/deletions
                        for line in diff_content.split('\\n'):
                            if line.startswith('+') and not line.startswith('+++'):
                                additions += 1
                            elif line.startswith('-') and not line.startswith('---'):
                                deletions += 1
                                
            except Exception as e:
                self.logger.debug(f"Failed to get diff for commit {commit.hexsha}: {e}")
        
        # Get tags and branches containing this commit
        tags = []
        branches = []
        
        try:
            # This is expensive, so we'll skip it for now and compute it separately if needed
            pass
        except Exception as e:
            self.logger.debug(f"Failed to get refs for commit {commit.hexsha}: {e}")
        
        return CommitData(
            sha=commit.hexsha,
            message=commit.message.strip(),
            author_name=commit.author.name,
            author_email=commit.author.email,
            commit_date=datetime.fromtimestamp(commit.committed_date),
            files_changed=files_changed,
            additions=additions,
            deletions=deletions,
            diff=diff_text,
            parents=[p.hexsha for p in commit.parents],
            tags=tags,
            branches=branches
        )
    
    def extract_recent_commits(self, days: int = 30, batch_size: int = 1000) -> Iterator[List[CommitData]]:
        """
        Extract commits from the last N days.
        
        Args:
            days: Number of days to look back
            batch_size: Batch size for processing
        
        Yields:
            Batches of recent CommitData objects
        """
        since = datetime.now() - timedelta(days=days)
        return self.extract_commits(since=since, batch_size=batch_size)
    
    def extract_commits_by_author(self, author_email: str, max_count: int = 1000) -> List[CommitData]:
        """
        Extract commits by a specific author.
        
        Args:
            author_email: Author's email address
            max_count: Maximum number of commits to extract
        
        Returns:
            List of CommitData objects
        """
        commits = []
        for batch in self.extract_commits(max_count=max_count):
            for commit in batch:
                if commit.author_email == author_email:
                    commits.append(commit)
        return commits
    
    def extract_file_history(self, file_path: str, max_count: int = 100) -> List[CommitData]:
        """
        Extract commit history for a specific file.
        
        Args:
            file_path: Path to the file
            max_count: Maximum number of commits to extract
        
        Returns:
            List of CommitData objects affecting the file
        """
        commits = []
        for batch in self.extract_commits(paths=[file_path], max_count=max_count):
            commits.extend(batch)
        return commits
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the repository.
        
        Returns:
            Dictionary with repository statistics
        """
        with PerformanceLogger(self.logger, "repository stats calculation"):
            stats = {
                'total_commits': 0,
                'total_branches': 0,
                'total_tags': 0,
                'total_contributors': set(),
                'languages': {},
                'latest_commit_date': None,
                'oldest_commit_date': None
            }
            
            # Count branches and tags
            stats['total_branches'] = len(list(self.repo.branches))
            stats['total_tags'] = len(list(self.repo.tags))
            
            # Sample commits to get overall statistics
            sample_size = 10000
            commit_count = 0
            
            for commit in self.repo.iter_commits():
                commit_count += 1
                stats['total_contributors'].add(commit.author.email)
                
                commit_date = datetime.fromtimestamp(commit.committed_date)
                if stats['latest_commit_date'] is None or commit_date > stats['latest_commit_date']:
                    stats['latest_commit_date'] = commit_date
                if stats['oldest_commit_date'] is None or commit_date < stats['oldest_commit_date']:
                    stats['oldest_commit_date'] = commit_date
                
                if commit_count >= sample_size:
                    break
            
            stats['total_commits'] = commit_count
            stats['total_contributors'] = len(stats['total_contributors'])
            
            self.logger.info(f"Repository stats: {json.dumps(stats, default=str, indent=2)}")
            
            return stats
    
    def save_commits_to_cache(self, commits: List[CommitData], filename: str) -> None:
        """
        Save commits to cache file.
        
        Args:
            commits: List of CommitData objects
            filename: Cache filename
        """
        cache_file = self.cache_dir / f"{filename}.json"
        
        # Convert to serializable format
        serializable_commits = []
        for commit in commits:
            commit_dict = asdict(commit)
            commit_dict['commit_date'] = commit.commit_date.isoformat()
            serializable_commits.append(commit_dict)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_commits, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(commits)} commits to cache: {cache_file}")
    
    def load_commits_from_cache(self, filename: str) -> List[CommitData]:
        """
        Load commits from cache file.
        
        Args:
            filename: Cache filename
        
        Returns:
            List of CommitData objects
        """
        cache_file = self.cache_dir / f"{filename}.json"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            commit_dicts = json.load(f)
        
        commits = []
        for commit_dict in commit_dicts:
            # Convert date string back to datetime
            commit_dict['commit_date'] = datetime.fromisoformat(commit_dict['commit_date'])
            commits.append(CommitData(**commit_dict))
        
        self.logger.info(f"Loaded {len(commits)} commits from cache: {cache_file}")
        return commits
    
    def export_to_dataframe(self, commits: List[CommitData]) -> pd.DataFrame:
        """
        Export commits to a pandas DataFrame for analysis.
        
        Args:
            commits: List of CommitData objects
        
        Returns:
            pandas DataFrame with commit data
        """
        data = []
        for commit in commits:
            row = {
                'sha': commit.sha,
                'message': commit.message,
                'author_name': commit.author_name,
                'author_email': commit.author_email,
                'commit_date': commit.commit_date,
                'num_files_changed': len(commit.files_changed),
                'additions': commit.additions,
                'deletions': commit.deletions,
                'total_changes': commit.additions + commit.deletions,
                'num_parents': len(commit.parents),
                'is_merge': len(commit.parents) > 1,
                'message_length': len(commit.message),
                'files_changed': ','.join(commit.files_changed[:10])  # First 10 files
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Exported {len(commits)} commits to DataFrame with shape {df.shape}")
        return df