import yaml
from ui.main import execute_video_pipeline

if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    execute_video_pipeline(config)