import json
import os
import subprocess
from pathlib import Path

def clone_wlasl_repo(temp_dir):
    print(f"Cloning WLASL repo into {temp_dir}...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/dxli94/WLASL.git", temp_dir], check=True)

def filter_wlasl100(full_json_path, output_path):
    print(f"Filtering WLASL100 from {full_json_path}...")
    with open(full_json_path, 'r') as f:
        data = json.load(f)
    
    # Sort glosses by number of instances
    data.sort(key=lambda x: len(x['instances']), reverse=True)
    
    # Take top 100
    wlasl100 = data[:100]
    
    with open(output_path, 'w') as f:
        json.dump(wlasl100, f, indent=2)
    
    print(f"Saved WLASL100 manifest to {output_path}")
    return len(wlasl100)

if __name__ == "__main__":
    temp_repo_dir = Path("temp_wlasl_repo")
    output_manifest = Path("data/wlasl100_manifest.json")
    
    if not temp_repo_dir.exists():
        clone_wlasl_repo(str(temp_repo_dir))
    
    json_path = temp_repo_dir / "start_kit" / "WLASL_v0.3.json"
    if json_path.exists():
        count = filter_wlasl100(str(json_path), str(output_manifest))
        print(f"Successfully processed {count} glosses.")
    else:
        print(f"Error: {json_path} not found in cloned repo.")
