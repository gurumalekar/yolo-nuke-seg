from pathlib import Path

def create_output_dir(base_name, dims, depth):
    output_dir = Path(f"{base_name}_dims{dims}_depth{depth}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir