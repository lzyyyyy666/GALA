"""Configuration Manager"""
import os.path
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Experiment Configuration Data Class"""
    data_path: str
    output_dir: str
    model_name: str
    repo_path: Optional[str] = None
    timestamp: Optional[str] = None
    git_commit: Optional[str] = None
    command_line: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization Processing"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def save_experiment_config(
        arguments,
        output_dir: Path,
        filename: str = "experiment_config.yml"
    ) -> Path:
        """
        Save experiment configuration to YAML file
        
        Args:
            config: Experiment configuration object
            output_dir: Output directory
            filename: Configuration file name
            
        Returns:
            Path: Configuration file path
        """
        config_file = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to dictionary and save
        # config_dict = asdict(config)
        config_dict = vars(arguments)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        return config_file
    
    @staticmethod
    def load_experiment_config(config_path: Path) -> ExperimentConfig:
        """
        Load experiment configuration from YAML file
        
        Args:
            config_path: Configuration file path
            
        Returns:
            ExperimentConfig: Experiment configuration object
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return ExperimentConfig(**config_dict)
    
    @staticmethod
    def create_from_args(args, **kwargs) -> ExperimentConfig:
        """
        Create configuration from command line arguments
        
        Args:
            args: Command line argument object
            **kwargs: Additional arguments
            
        Returns:
            ExperimentConfig: Experiment configuration object
        """
        # Build command line string
        command_line = "python main.py " + " ".join([
            f"--data_path {args.data_path}",
            f"--output_dir {args.output_dir}",
            f"--model_name {args.model_name}"
        ])
        
        # Add additional parameters
        if hasattr(args, 'repo_path') and args.repo_path:
            command_line += f" --repo_path {args.repo_path}"
        
        return ExperimentConfig(
            data_path=str(args.data_path),
            output_dir=str(args.output_dir),
            model_name=args.model_name,
            repo_path=kwargs.get('repo_path'),
            command_line=command_line
        )