from datasets import load_dataset
ds = load_dataset("Zihan1004/FNSPID", split='train', streaming=True)
print("Keys:", next(iter(ds)).keys())

print("Scanning first 100 items for symbols...")
symbols = set()
for i, item in enumerate(ds):
    if i > 100: break
    symbols.add(item.get('Stock_symbol'))
    
print("Sample Symbols:", symbols)
