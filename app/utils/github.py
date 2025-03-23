"""GitHub scraper utility for Vanna AI Repository Q&A System."""

import os
import tempfile
from git import Repo
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class GitHubScraper:
    """Scraper for GitHub repositories."""
    
    def __init__(self, repo_url):
        self.repo_url = repo_url
        self.temp_dir = None
        
    def clone_repository(self):
        """Clone the GitHub repository to a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Cloning repository {self.repo_url} to {self.temp_dir}")
        
        try:
            Repo.clone_from(self.repo_url, self.temp_dir)
            return self.temp_dir
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
    
    def get_file_paths(self, extensions=None):
        """Get all file paths from the cloned repository."""
        if not self.temp_dir:
            self.clone_repository()
        
        file_paths = []
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                # Skip hidden files and directories
                if file.startswith('.') or '/.git/' in root:
                    continue
                    
                # Filter by extensions if provided
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue
                    
                file_paths.append(os.path.join(root, file))
        
        return file_paths
    
    def cleanup(self):
        """Remove the temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            logger.info("Temporary directory cleaned up.")
