from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
ARTICLES=['2026-11-02/01_substack_monday_overnight/experiment.py', '2026-11-02/02_substack_tuesday_ml_rotation/experiment.py', '2026-11-02/03_substack_thursday_buffer/experiment.py', '2026-11-02/04_substack_friday_ulcer/experiment.py', '2026-11-02/05_medium_monday_rebound/experiment.py', '2026-11-02/06_medium_wednesday_inside_nr7/experiment.py', '2026-11-02/07_medium_friday_ichimoku/experiment.py']

def main():
    parser=argparse.ArgumentParser(description="Run an article's complete research experiment")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--article",type=int,choices=range(1,8))
    group.add_argument("--all",action="store_true")
    args=parser.parse_args()
    selected=ARTICLES if args.all else [ARTICLES[args.article-1]]
    for article in selected:
        print("Running",article,flush=True)
        subprocess.run([sys.executable,str(ROOT/article)],check=True)

if __name__=="__main__":main()
