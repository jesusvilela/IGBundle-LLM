import json
import os
import glob

def main():
    stats = {}
    
    # 1. Load ARC Results
    if os.path.exists("arc_evaluation_results.json"):
        try:
            with open("arc_evaluation_results.json", "r") as f:
                arc_data = json.load(f)
                stats["accuracy_igbundle"] = arc_data.get("summary", {}).get("accuracy", 0.0)
                stats["mfr_compliance"] = arc_data.get("summary", {}).get("mfr_compliance", 0.0)
                print(f"Loaded ARC stats: Acc={stats['accuracy_igbundle']}, MFR={stats['mfr_compliance']}")
        except Exception as e:
            print(f"Error loading ARC results: {e}")
            
    # 2. Load Arena Results (Sigma)
    if os.path.exists("arena_answers.json"):
        try:
            with open("arena_answers.json", "r") as f:
                arena_data = json.load(f)
                # Calculate average sigma
                sigmas = [x.get("sigma", 0) for x in arena_data if "sigma" in x]
                if sigmas:
                    avg_sigma = sum(sigmas) / len(sigmas)
                    stats["curvature_sigma"] = avg_sigma
                    print(f"Loaded Arena stats: Sigma={avg_sigma}")
        except Exception as e:
            print(f"Error loading Arena results: {e}")
            
    # 3. Update thesis_stats.json
    thesis_stats_path = "thesis_stats.json"
    current_stats = {}
    if os.path.exists(thesis_stats_path):
        try:
            with open(thesis_stats_path, "r") as f:
                current_stats = json.load(f)
        except:
            pass
            
    # Merge
    current_stats.update(stats)
    
    with open(thesis_stats_path, "w") as f:
        json.dump(current_stats, f, indent=2)
    print(f"Updated {thesis_stats_path} with {stats}")

if __name__ == "__main__":
    main()
