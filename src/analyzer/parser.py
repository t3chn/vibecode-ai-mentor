"""Tree-sitter based Python code parser for extracting AST elements."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tree_sitter
import tree_sitter_python as tspython
from aiofiles import open as aio_open

# Make module work without full config
try:
    from src.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


class PythonParser:
    """Parser for Python code using tree-sitter."""

    def __init__(self) -> None:
        """Initialize the parser with Python language."""
        self.language = tree_sitter.Language(tspython.language())
        self.parser = tree_sitter.Parser(self.language)
        self._code: Optional[bytes] = None
        self._tree: Optional[tree_sitter.Tree] = None

    async def parse_file(self, file_path: Path | str) -> tree_sitter.Tree:
        """Parse a Python file asynchronously.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Parsed tree
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: For parsing errors
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            async with aio_open(file_path, "rb") as f:
                content = await f.read()
            return self.parse_code(content.decode("utf-8"))
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise

    def parse_code(self, code_str: str) -> tree_sitter.Tree:
        """Parse Python code string.
        
        Args:
            code_str: Python code as string
            
        Returns:
            Parsed tree
        """
        self._code = code_str.encode("utf-8")
        self._tree = self.parser.parse(self._code)
        return self._tree

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract all function definitions from parsed code.
        
        Returns:
            List of function info dictionaries
        """
        if not self._tree:
            return []
            
        functions = []
        self._extract_functions_recursive(self._tree.root_node, functions)
        return functions

    def _extract_functions_recursive(
        self, node: tree_sitter.Node, functions: List[Dict[str, Any]]
    ) -> None:
        """Recursively extract function definitions."""
        if node.type == "function_definition":
            func_info = {
                "name": self._get_function_name(node),
                "parameters": self._get_function_params(node),
                "return_type": self._get_return_type(node),
                "docstring": self._get_docstring(node),
                "decorators": self._get_decorators(node),
                "location": self.get_node_location(node),
                "text": self.get_node_text(node),
            }
            functions.append(func_info)
            
        for child in node.children:
            self._extract_functions_recursive(child, functions)

    def extract_classes(self) -> List[Dict[str, Any]]:
        """Extract all class definitions from parsed code.
        
        Returns:
            List of class info dictionaries
        """
        if not self._tree:
            return []
            
        classes = []
        self._extract_classes_recursive(self._tree.root_node, classes)
        return classes

    def _extract_classes_recursive(
        self, node: tree_sitter.Node, classes: List[Dict[str, Any]]
    ) -> None:
        """Recursively extract class definitions."""
        if node.type == "class_definition":
            class_info = {
                "name": self._get_class_name(node),
                "bases": self._get_class_bases(node),
                "docstring": self._get_docstring(node),
                "decorators": self._get_decorators(node),
                "methods": self._get_class_methods(node),
                "attributes": self._get_class_attributes(node),
                "location": self.get_node_location(node),
                "text": self.get_node_text(node),
            }
            classes.append(class_info)
            # Don't recurse into the class itself
            return
            
        for child in node.children:
            self._extract_classes_recursive(child, classes)

    def extract_imports(self) -> List[Dict[str, Any]]:
        """Extract all import statements from parsed code.
        
        Returns:
            List of import info dictionaries
        """
        if not self._tree:
            return []
            
        imports = []
        self._extract_imports_recursive(self._tree.root_node, imports)
        return imports

    def extract_global_variables(self) -> List[Dict[str, Any]]:
        """Extract all global variable assignments from parsed code.
        
        Returns:
            List of global variable info dictionaries
        """
        if not self._tree:
            return []
            
        globals_vars = []
        # Only look at top-level assignments
        for child in self._tree.root_node.children:
            if child.type == "expression_statement":
                expr = child.children[0] if child.children else None
                if expr and expr.type == "assignment":
                    var_info = {
                        "name": self._get_assignment_target(expr),
                        "value": self._get_assignment_value(expr),
                        "location": self.get_node_location(child),
                        "text": self.get_node_text(child),
                    }
                    if var_info["name"]:  # Only add if we got a valid name
                        globals_vars.append(var_info)
        return globals_vars

    def extract_comments(self) -> List[Dict[str, Any]]:
        """Extract all comments from parsed code.
        
        Returns:
            List of comment info dictionaries
        """
        if not self._tree:
            return []
            
        comments = []
        self._extract_comments_recursive(self._tree.root_node, comments)
        return comments

    def _extract_imports_recursive(
        self, node: tree_sitter.Node, imports: List[Dict[str, Any]]
    ) -> None:
        """Recursively extract import statements."""
        if node.type in ("import_statement", "import_from_statement"):
            import_info = {
                "type": node.type,
                "module": self._get_import_module(node),
                "names": self._get_import_names(node),
                "location": self.get_node_location(node),
                "text": self.get_node_text(node),
            }
            imports.append(import_info)
            
        for child in node.children:
            self._extract_imports_recursive(child, imports)

    def _extract_comments_recursive(
        self, node: tree_sitter.Node, comments: List[Dict[str, Any]]
    ) -> None:
        """Recursively extract comments."""
        if node.type == "comment":
            comment_info = {
                "text": self.get_node_text(node).lstrip("#").strip(),
                "location": self.get_node_location(node),
                "raw_text": self.get_node_text(node),
            }
            comments.append(comment_info)
            
        for child in node.children:
            self._extract_comments_recursive(child, comments)

    def get_node_text(self, node: tree_sitter.Node) -> str:
        """Get source text for an AST node.
        
        Args:
            node: Tree-sitter node
            
        Returns:
            Source text of the node
        """
        if not self._code:
            return ""
        return self._code[node.start_byte:node.end_byte].decode("utf-8")

    def get_node_location(self, node: tree_sitter.Node) -> Dict[str, int]:
        """Get line numbers for a node.
        
        Args:
            node: Tree-sitter node
            
        Returns:
            Dictionary with start_line and end_line
        """
        return {
            "start_line": node.start_point[0] + 1,  # Convert to 1-based
            "end_line": node.end_point[0] + 1,
            "start_column": node.start_point[1],
            "end_column": node.end_point[1],
        }

    # Helper methods for extracting specific information

    def _get_function_name(self, node: tree_sitter.Node) -> str:
        """Extract function name from function definition node."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_text(child)
        return ""

    def _get_function_params(self, node: tree_sitter.Node) -> List[Dict[str, str]]:
        """Extract function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameters":
                for param in child.children:
                    if param.type == "identifier":
                        params.append({
                            "name": self.get_node_text(param),
                            "type": None,
                        })
                    elif param.type == "typed_parameter":
                        name_node = param.child_by_field_name("name")
                        type_node = param.child_by_field_name("type")
                        params.append({
                            "name": self.get_node_text(name_node) if name_node else "",
                            "type": self.get_node_text(type_node) if type_node else None,
                        })
        return params

    def _get_return_type(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract return type annotation from function."""
        for child in node.children:
            if child.type == "type":
                return self.get_node_text(child)
        return None

    def _get_docstring(self, node: tree_sitter.Node) -> Optional[str]:
        """Extract docstring from function or class."""
        body_node = node.child_by_field_name("body")
        if body_node and body_node.children:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0]
                if expr.type == "string":
                    docstring = self.get_node_text(expr)
                    # Remove quotes
                    return docstring.strip("\"'")
        return None

    def _get_decorators(self, node: tree_sitter.Node) -> List[str]:
        """Extract decorators from function or class."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorators.append(self.get_node_text(child).lstrip("@"))
        return decorators

    def _get_class_name(self, node: tree_sitter.Node) -> str:
        """Extract class name from class definition node."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_text(child)
        return ""

    def _get_class_bases(self, node: tree_sitter.Node) -> List[str]:
        """Extract base classes from class definition."""
        bases = []
        for child in node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type == "identifier":
                        bases.append(self.get_node_text(arg))
        return bases

    def _get_class_methods(self, node: tree_sitter.Node) -> List[str]:
        """Extract method names from class definition."""
        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    methods.append(self._get_function_name(child))
        return methods

    def _get_class_attributes(self, node: tree_sitter.Node) -> List[str]:
        """Extract class attributes (simple assignments in __init__)."""
        attributes = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition" and self._get_function_name(child) == "__init__":
                    init_body = child.child_by_field_name("body")
                    if init_body:
                        for stmt in init_body.children:
                            if stmt.type == "expression_statement":
                                expr = stmt.children[0]
                                if expr.type == "assignment":
                                    left = expr.child_by_field_name("left")
                                    if left and left.type == "attribute":
                                        attr_name = left.child_by_field_name("attribute")
                                        if attr_name:
                                            attributes.append(self.get_node_text(attr_name))
        return attributes

    def _get_import_module(self, node: tree_sitter.Node) -> str:
        """Extract module name from import statement."""
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    return self.get_node_text(child)
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node:
                return self.get_node_text(module_node)
        return ""

    def _get_import_names(self, node: tree_sitter.Node) -> List[str]:
        """Extract imported names from import statement."""
        names = []
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    names.append(self.get_node_text(child))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        names.append(self.get_node_text(name_node))
        elif node.type == "import_from_statement":
            for child in node.children:
                if child.type == "import_from_names":
                    for name_child in child.children:
                        if name_child.type == "dotted_name" or name_child.type == "identifier":
                            names.append(self.get_node_text(name_child))
        return names

    def _get_assignment_target(self, node: tree_sitter.Node) -> str:
        """Extract target variable name from assignment."""
        left = node.child_by_field_name("left")
        if left:
            if left.type == "identifier":
                return self.get_node_text(left)
            elif left.type == "attribute":
                # For now, skip attribute assignments like obj.attr = value
                return ""
        return ""

    def _get_assignment_value(self, node: tree_sitter.Node) -> str:
        """Extract value from assignment (as text)."""
        right = node.child_by_field_name("right")
        if right:
            return self.get_node_text(right)
        return ""