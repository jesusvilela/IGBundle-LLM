from datasets import load_dataset
print("Loading dataset...")
dataset = load_dataset("derek-thomas/ScienceQA", split="train", streaming=True)

def is_physics_vision(example):
    return example['image'] is not None and example['topic'] in ['natural science', 'physics']

print("Filtering...")
dataset = dataset.filter(is_physics_vision)
print("Peeking at first item...")
item = next(iter(dataset))
print("Image Type:", type(item['image']))
print("Image Content:", item['image'])
