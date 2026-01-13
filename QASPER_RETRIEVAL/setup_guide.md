# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy
pip install sentence-transformers
pip install rank-bm25
pip install tqdm
pip install matplotlib seaborn
pip install scipy


numpy>=1.21.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2
tqdm>=4.65.0
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
torch>=2.0.0  # Required by sentence-transformers


# Download from official repository
curl -L "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-dev-v0.3.json" -o qasper-dev-v0.3.json

# Or download manually from:
# https://github.com/allenai/qasper

# Basic evaluation (Tier 1 only)
python qasper_implementation.py
Dataset Info:

Train set: 2,593 papers, 20,396 questions
Dev set: 281 papers, 2,190 questions
Test set: 281 papers, 2,166 questions
Average paper length: ~5,000 tokens (NLP research papers)



# Basic evaluation (Tier 1 only)
python qasper_implementation.py

# Full comparison (all methods)
python baseline_comparison.py


project/
├── qasper_implementation.py      # Main implementation
├── baseline_comparison.py        # Comparison & ablation
├── qasper-dev-v0.3.json         # Dataset (download)
├── requirements.txt              # Dependencies
├── results/
│   ├── qasper_tier1_results.json
│   ├── comparison_results.json
│   └── retrieval_comparison.png
└── README.md


from qasper_implementation import run_qasper_evaluation

# Evaluate on 10 papers (quick test)
results = run_qasper_evaluation(
    qasper_path="qasper-dev-v0.3.json",
    num_papers=10,
    output_path="results_quick.json"
)

# Full evaluation on all dev papers
results = run_qasper_evaluation(
    qasper_path="qasper-dev-v0.3.json",
    num_papers=281,  # All dev papers
    output_path="results_full.json"
)


from baseline_comparison import run_comprehensive_comparison

# Compare all methods
results = run_comprehensive_comparison(
    qasper_path="qasper-dev-v0.3.json",
    num_papers=20
)

# Expected output:
# - Comparison table printed to console
# - comparison_results.json saved
# - retrieval_comparison.png generated


from baseline_comparison import compute_statistical_significance

# Run paired t-test
compute_statistical_significance(
    qasper_path="qasper-dev-v0.3.json",
    num_papers=20
)

# Tests if improvement over baseline is statistically significant


