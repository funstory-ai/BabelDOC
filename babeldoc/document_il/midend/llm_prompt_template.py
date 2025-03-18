import logging
import re
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class LLMPromptTemplate:
    """Handles loading and parsing LLM prompt templates from XML."""
    
    def __init__(self, template_path: str | Path, translation_config=None):
        """Initialize template with XML file path.
        
        Args:
            template_path: Path to XML template file
            translation_config: Translation configuration for debug settings
        """
        self.template_path = Path(template_path)
        self.translation_config = translation_config

    @classmethod
    def from_xml_file(cls, path: str | Path, translation_config=None) -> "LLMPromptTemplate":
        """Load template from XML file.
        
        Args:
            path: Path to XML template file
            
        Returns:
            Initialized LLMPromptTemplate instance
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ET.ParseError: If XML is malformed
            ValueError: If required template sections are missing
        """
        # Verify XML file exists and is valid
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Basic validation
            if root.tag != "TranslationTask":
                raise ValueError("Root element must be TranslationTask")
            
            # Return instance with validated path
            return cls(path, translation_config)
            
        except (FileNotFoundError, ET.ParseError) as e:
            logger.error(f"Failed to load template from {path}: {e}")
            raise

    def render(self, variables: dict[str, str]) -> str:
        """Render template with variable substitution, keeping XML structure.
        
        Args:
            variables: Dictionary of variable names and values
            
        Returns:
            XML string with variables replaced
        """
        if self.translation_config and self.translation_config.debug:
            logger.info("="*80)
            logger.info("Template Processing Start")
            logger.info("="*80)
            
            # Log original template content
            logger.info("Original Template:")
            logger.info("-"*80)
            with open(self.template_path, 'r', encoding='utf-8') as f:
                logger.info(f.read())
            logger.info("-"*80)
            
            # Log variables
            logger.info("Variables:")
            for key, value in variables.items():
                logger.info(f"{key}: {value}")
            logger.info("-"*80)
            
        tree = ET.parse(self.template_path)
        root = tree.getroot()
        
        # Process all nodes with variable substitution and conditional handling
        self._process_node(root, variables)
        
        # Generate final result
        result = ET.tostring(root, encoding='unicode', method='xml')
        
        if self.translation_config and self.translation_config.debug:
            logger.info("Processed Template:")
            logger.info("-"*80)
            logger.info(result)
            logger.info("="*80)
            
        return result  # Return full XML with substituted variables and processed conditions

    def _process_node(self, node: ET.Element, variables: dict[str, str]):
        """Recursively process XML nodes for variable substitution and conditionals.
        
        Args:
            node: Current XML node
            variables: Dictionary of variable values
        """
        # Process process sections first to handle conditions
        if node.tag == "Process":
            # Get current content type
            content_type = variables.get("contentType", "")
            
            # Determine which condition to keep
            target_condition = ""
            if content_type == "title":
                target_condition = "if_title"
            elif content_type.endswith("caption"):
                target_condition = "if_caption"
            elif content_type == "toc_entry":
                target_condition = "if_toc_entry"
            else:
                target_condition = "if_paragraph"
            
            # Remove non-matching conditional sections
            children = list(node)
            for child in children:
                if child.tag.startswith("if_"):
                    if child.tag != target_condition:
                        node.remove(child)
                    else:
                        # Keep matched condition content but remove wrapper
                        for subchild in list(child):
                            node.append(subchild)
                        node.remove(child)
        
        # Handle text nodes
        if node.text and isinstance(node.text, str):
            node.text = self._substitute_variables(node.text.strip(), variables)
        if node.tail and isinstance(node.tail, str):
            node.tail = self._substitute_variables(node.tail.strip(), variables)
        
        # Process remaining children
        for child in list(node):
            if not child.tag.startswith("if_"):  # Skip already processed conditions
                self._process_node(child, variables)
            
    def _evaluate_condition(self, condition: str, variables: dict[str, str]) -> bool:
        """Evaluate a conditional section.
        
        Args:
            condition: Name of condition (e.g., "title" from ${if_title})
            variables: Dictionary of variable values
            
        Returns:
            True if condition is met, False otherwise
        """
        result = False
        content_type = variables.get("contentType", "")
        
        if condition == "title":
            result = (content_type == "title")
        elif condition == "caption":
            result = content_type.endswith("caption")
        elif condition == "paragraph":
            result = (content_type == "plain text")
        elif condition == "toc_entry":
            result = (content_type == "toc_entry")
            
        if self.translation_config and self.translation_config.debug:
            logger.info(f"Evaluating condition: {condition}")
            logger.info(f"Content type: {content_type}")
            logger.info(f"Result: {result}")
            logger.info("-"*40)
            
        return result

    def _substitute_variables(self, text: str, variables: dict[str, str]) -> str:
        """Replace ${var} placeholders with actual values.
        
        Args:
            text: Template text containing variables
            variables: Dictionary of variable names and values
            
        Returns:
            Text with variables replaced
        """
        def replace_var(match):
            var_name = match.group(1)
            return variables.get(var_name, f"${{{var_name}}}")
            
        return re.sub(r'\${([^}]+)}', replace_var, text)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"LLMPromptTemplate(template_path='{self.template_path}')"
