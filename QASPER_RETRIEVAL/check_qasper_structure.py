"""
Check the actual structure of your QASPER file
"""

import json

with open('qasper-dev-v0.3.json', 'r') as f:
    data = json.load(f)

# Get first paper
paper_id = list(data.keys())[0]
paper = data[paper_id]

print("=" * 70)
print("QASPER DATA STRUCTURE ANALYSIS")
print("=" * 70)

print(f"\nPaper ID: {paper_id}")
print(f"Title: {paper['title']}")

print("\n--- KEYS IN PAPER ---")
for key in paper.keys():
    print(f"  {key}: {type(paper[key])}")

print("\n--- ABSTRACT ---")
print(f"Type: {type(paper['abstract'])}")
print(f"Length: {len(paper['abstract']) if isinstance(paper['abstract'], str) else 'N/A'}")
print(f"Preview: {str(paper['abstract'])[:200]}...")

print("\n--- FULL_TEXT STRUCTURE ---")
full_text = paper['full_text']
print(f"Type: {type(full_text)}")
print(f"Length: {len(full_text)}")

if isinstance(full_text, list) and len(full_text) > 0:
    print(f"\nFirst element type: {type(full_text[0])}")
    print(f"First element: {full_text[0]}")
    
    if isinstance(full_text[0], dict):
        print("\nFormat: List of dicts")
        print(f"Keys in first section: {list(full_text[0].keys())}")
        for key, value in full_text[0].items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, list):
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
    
    elif isinstance(full_text[0], list):
        print("\nFormat: List of lists")
        print(f"First section length: {len(full_text[0])}")
        for i, item in enumerate(full_text[0]):
            print(f"  [{i}]: {type(item)}")
            if isinstance(item, str):
                print(f"       Preview: {item[:100]}...")
            elif isinstance(item, list):
                print(f"       Length: {len(item)}")

print("\n--- QAS STRUCTURE ---")
qas = paper['qas']
print(f"Number of questions: {len(qas)}")
if len(qas) > 0:
    print(f"\nFirst question structure:")
    q = qas[0]
    for key, value in q.items():
        print(f"  {key}: {type(value)}")
        if key == 'answers' and isinstance(value, list) and len(value) > 0:
            print(f"    First answer keys: {list(value[0].keys())}")

print("\n" + "=" * 70)