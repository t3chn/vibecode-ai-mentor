"""Test parser class extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.parser import PythonParser

test_code = '''
class DataProcessor:
    """Process various data formats."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = []
    
    def process_json(self, data: str) -> dict:
        """Process JSON data."""
        return {"processed": data}
'''

parser = PythonParser()
tree = parser.parse_code(test_code)
print("Tree root type:", tree.root_node.type)
print("\nRoot children:")
for i, child in enumerate(tree.root_node.children):
    print(f"  {i}: {child.type} at line {child.start_point[0] + 1}")
    if child.type == "class_definition":
        print(f"     Text preview: {parser.get_node_text(child)[:50]}...")

classes = parser.extract_classes()
print(f"\nExtracted {len(classes)} classes")
for cls in classes:
    print(f"  - {cls['name']} at lines {cls['location']['start_line']}-{cls['location']['end_line']}")
    print(f"    Methods: {cls['methods']}")