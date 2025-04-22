# src/permuto/permuto.py
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import copy
import re
from contextlib import contextmanager

try:
    import jsonpointer
except ImportError:
    raise ImportError("The 'jsonpointer' library is required for permuto. Please install it: pip install jsonpointer")


from .exceptions import (
    PermutoException,
    PermutoInvalidOptionsError,
    PermutoCycleError,
    PermutoMissingKeyError,
    PermutoReverseError,
)

# Define Json types for type hinting
JsonPrimitive = Union[str, int, float, bool, None]
JsonType = Union[JsonPrimitive, List[Any], Dict[str, Any]]
ContextType = Dict[str, Any]
TemplateType = JsonType

# --- Sentinel Object for Lookup Failure ---
_NOT_FOUND = object()


class Options:
    """Configuration options for Permuto processing."""
    def __init__(
        self,
        variable_start_marker: str = "${",
        variable_end_marker: str = "}",
        on_missing_key: str = "ignore",  # 'ignore' or 'error'
        enable_string_interpolation: bool = False,
    ):
        self.variable_start_marker = variable_start_marker
        self.variable_end_marker = variable_end_marker
        self.on_missing_key = on_missing_key
        self.enable_string_interpolation = enable_string_interpolation
        self.validate()

    def validate(self) -> None:
        """Validates the options."""
        if not self.variable_start_marker:
            raise PermutoInvalidOptionsError("variable_start_marker cannot be empty.")
        if not self.variable_end_marker:
            raise PermutoInvalidOptionsError("variable_end_marker cannot be empty.")
        if self.variable_start_marker == self.variable_end_marker:
            raise PermutoInvalidOptionsError("variable_start_marker and variable_end_marker cannot be identical.")
        if self.on_missing_key not in ["ignore", "error"]:
            raise PermutoInvalidOptionsError("on_missing_key must be 'ignore' or 'error'.")

# --- Helper Functions ---

def _stringify_json(value: JsonType) -> str:
    """Converts a resolved JSON value to its string representation for interpolation."""
    if isinstance(value, str):
        return value
    elif value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return json.dumps(value)
    elif isinstance(value, (list, dict)):
        return json.dumps(value, separators=(',', ':'))
    else:
        return ""

@contextmanager
def _active_path_guard(active_paths: Set[str], path_to_check: str):
    """RAII-like guard for cycle detection using a context manager."""
    if not path_to_check:
        yield
        return

    if path_to_check in active_paths:
        cycle_info = f"path '{path_to_check}' already being processed"
        raise PermutoCycleError("Cycle detected during substitution", cycle_info)

    active_paths.add(path_to_check)
    try:
        yield
    finally:
        active_paths.remove(path_to_check)

def _resolve_path(
    context: ContextType,
    path: str,
    options: Options,
    full_placeholder_for_error: str
) -> Union[JsonType, object]:
    if not path:
        if options.on_missing_key == "error":
            raise PermutoMissingKeyError("Path cannot be empty within placeholder", path)
        return _NOT_FOUND

    current_obj = context
    segments = path.split('.')
    seg_idx = 0
    current_path_for_error = [] # Track path for error reporting

    try:
        while seg_idx < len(segments):
            # --- Try multi-part keys first (longest to shortest containing current segment) ---
            found_match = False
            # Check combinations from current index up to the end
            for lookahead in range(len(segments) - seg_idx - 1, -1, -1): # iterate lookahead from max down to 0
                potential_key = ".".join(segments[seg_idx : seg_idx + 1 + lookahead])

                if isinstance(current_obj, dict) and potential_key in current_obj:
                    # Found a key (could be single or multi-part)
                    current_obj = current_obj[potential_key]
                    # Update error path tracking - conceptually replace segments with the key found
                    # For simplicity in tracking, just add the found key. Precise segment replacement is tricky.
                    current_path_for_error.append(potential_key) # Track the key we actually used
                    # Advance index past all consumed segments
                    seg_idx += (1 + lookahead)
                    found_match = True
                    break # Exit lookahead loop, continue main segment loop
                elif isinstance(current_obj, list) and potential_key.isdigit() and lookahead == 0:
                    # Only consider single segment for list index
                    try:
                        current_obj = current_obj[int(potential_key)]
                        current_path_for_error.append(potential_key)
                        seg_idx += 1
                        found_match = True
                        break # Exit lookahead loop
                    except IndexError:
                        # Index out of bounds, treat as lookup failure below
                         pass # Allow loop to finish and raise KeyError


            if found_match:
                continue # Continue main segment loop

            # --- If no match found (single, multi, or list index) ---
            # Raise error based on the first segment we tried at this position
            first_segment_tried = segments[seg_idx]
            current_path_for_error.append(first_segment_tried) # Add the segment that failed
            raise KeyError(first_segment_tried)


        # Loop finished successfully
        return current_obj

    except (KeyError, IndexError, TypeError) as e:
        # Path traversal failed at some point
        failed_path_str = ".".join(current_path_for_error) # Use tracked path
        if options.on_missing_key == "error":
            raise PermutoMissingKeyError(
                 f"Key or path not found in context: '{path}' (failed near path '{failed_path_str}')", # Adjusted error message
                 path
             ) from e
        return _NOT_FOUND # Use sentinel for failure
    except Exception as e:
        if options.on_missing_key == "error":
            raise PermutoException(f"Unexpected error resolving path '{path}': {e}") from e
        return _NOT_FOUND # Use sentinel for failure

def _resolve_and_process_placeholder(
    path: str,
    full_placeholder: str,
    context: ContextType,
    options: Options,
    active_paths: Set[str]
) -> JsonType:
    """
    Resolves a placeholder path, handles cycles and missing keys,
    and recursively processes the result if it's a string.
    """
    with _active_path_guard(active_paths, path):
        resolved_value = _resolve_path(context, path, options, full_placeholder)

        # Check specifically for the _NOT_FOUND sentinel
        if resolved_value is _NOT_FOUND:
            # Lookup failed or invalid path ignored
            return full_placeholder
        else:
            # Successfully resolved (value could be None)
            if isinstance(resolved_value, str):
                return _process_string(resolved_value, context, options, active_paths)
            else:
                return resolved_value # Return actual value (including None)


def _process_string(
    template_str: str,
    context: ContextType,
    options: Options,
    active_paths: Set[str]
) -> JsonType:
    """
    Processes a string node, handling exact matches and interpolation based on options.
    """
    start_marker = options.variable_start_marker
    end_marker = options.variable_end_marker
    start_len = len(start_marker)
    end_len = len(end_marker)
    str_len = len(template_str)
    min_placeholder_len = start_len + end_len + 1

    # --- 1. Check for Exact Match Placeholder ---
    is_exact_match = False
    path = ""
    if (str_len >= min_placeholder_len and
        template_str.startswith(start_marker) and
        template_str.endswith(end_marker)):
        content = template_str[start_len:-end_len]
        # Ensure content does not contain markers that would break the exact match assumption
        if start_marker not in content and end_marker not in content:
             path = content
             if path: # Path cannot be empty
                is_exact_match = True

    if is_exact_match:
        return _resolve_and_process_placeholder(path, template_str, context, options, active_paths)

    # --- 2. Handle based on Interpolation Option ---
    if not options.enable_string_interpolation:
        return template_str

    # --- 3. Perform Interpolation ---
    result_parts: List[str] = []
    current_pos = 0
    while current_pos < str_len:
        start_pos = template_str.find(start_marker, current_pos)

        if start_pos == -1:
            result_parts.append(template_str[current_pos:])
            break

        result_parts.append(template_str[current_pos:start_pos])
        end_pos = template_str.find(end_marker, start_pos + start_len)

        if end_pos == -1:
            result_parts.append(template_str[start_pos:])
            break

        placeholder_path = template_str[start_pos + start_len : end_pos]
        full_placeholder = template_str[start_pos : end_pos + end_len]

        if not placeholder_path:
            result_parts.append(full_placeholder)
        else:
            resolved_value = _resolve_and_process_placeholder(
                placeholder_path, full_placeholder, context, options, active_paths
            )
            result_parts.append(_stringify_json(resolved_value))

        current_pos = end_pos + end_len

    return "".join(result_parts)


def _process_node(
    node: TemplateType,
    context: ContextType,
    options: Options,
    active_paths: Set[str]
) -> JsonType:
    """Recursively processes a node in the template."""
    if isinstance(node, dict):
        result_obj = {}
        for key, val in node.items():
            result_obj[key] = _process_node(val, context, options, active_paths)
        return result_obj
    elif isinstance(node, list):
        return [_process_node(element, context, options, active_paths) for element in node]
    elif isinstance(node, str):
        return _process_string(node, context, options, active_paths)
    else:
        return node


# --- Reverse Template Helpers ---

def _insert_pointer_at_context_path(
    reverse_template_node: Dict[str, Any],
    context_path: str, # Dot notation path for context structure
    pointer_to_insert: str # JSON Pointer string for result location
) -> None:
    """
    Inserts the result pointer into the nested reverse template structure,
    using simple dot splitting to define the structure.
    """
    if not context_path:
         raise PermutoReverseError("[Internal] Invalid empty context path.")

    # Basic path validation for empty segments
    if context_path.startswith('.') or context_path.endswith('.') or '..' in context_path:
        raise PermutoReverseError(f"[Internal] Invalid context path format: {context_path}")

    segments = context_path.split('.')
    current_node = reverse_template_node

    for i, segment in enumerate(segments):
        if not segment: # Check again after split
            raise PermutoReverseError(f"[Internal] Invalid context path format (empty segment): {context_path}")

        is_last_segment = (i == len(segments) - 1)

        if is_last_segment:
            # Assign the pointer string at the final key (segment), overwriting if needed.
            current_node[segment] = pointer_to_insert
        else:
            # Need to descend. Get or create the next node.
            existing_val = current_node.get(segment)
            if not isinstance(existing_val, dict):
                # If it doesn't exist, or is not a dict, overwrite/create a dict.
                current_node[segment] = {}
            # Descend into the object node.
            current_node = current_node[segment]

def _build_reverse_template_recursive(
    current_template_node: TemplateType,
    current_result_pointer_str: str, # JSON Pointer string for result location
    reverse_template_ref: Dict[str, Any], # The root reverse template being built
    options: Options
) -> None:
    """Recursively builds the reverse template."""

    if isinstance(current_template_node, dict):
        for key, val in current_template_node.items():
            escaped_key = jsonpointer.escape(key)
            next_pointer_str = f"{current_result_pointer_str}/{escaped_key}"
            _build_reverse_template_recursive(val, next_pointer_str, reverse_template_ref, options)
    elif isinstance(current_template_node, list):
        for i, element in enumerate(current_template_node):
            next_pointer_str = f"{current_result_pointer_str}/{i}"
            _build_reverse_template_recursive(element, next_pointer_str, reverse_template_ref, options)
    elif isinstance(current_template_node, str):
        template_str = current_template_node
        start_marker = options.variable_start_marker
        end_marker = options.variable_end_marker
        start_len = len(start_marker)
        end_len = len(end_marker)
        str_len = len(template_str)
        min_placeholder_len = start_len + end_len + 1

        is_exact_match = False
        context_path = ""
        if (str_len >= min_placeholder_len and
            template_str.startswith(start_marker) and
            template_str.endswith(end_marker)):
            content = template_str[start_len:-end_len]
            # Validate content doesn't contain markers AND path format is simple
            if start_marker not in content and end_marker not in content:
                context_path = content
                # Basic path validation (no empty segments, etc.)
                if context_path and '.' not in context_path[-1:] and '.' not in context_path[:1] and '..' not in context_path:
                    is_exact_match = True

        if is_exact_match:
            try:
                _insert_pointer_at_context_path(
                    reverse_template_ref, context_path, current_result_pointer_str
                )
            except PermutoReverseError as e:
                 raise PermutoReverseError(f"Error building reverse template for context path '{context_path}': {e}") from e
            except Exception as e:
                 raise PermutoReverseError(f"Unexpected error inserting pointer for context path '{context_path}': {e}") from e


def _reconstruct_context_recursive(
    reverse_node: JsonType,
    result_json: JsonType,
    parent_context_node: Union[Dict, List], # The node being built (dict or list)
    current_key_or_index: Union[str, int] # Key/Index in the parent to assign to
) -> None:
    """Recursively reconstructs the context using the reverse template."""

    if isinstance(reverse_node, dict):
        new_node: Dict[str, Any] = {}
        if isinstance(parent_context_node, dict):
            parent_context_node[current_key_or_index] = new_node
        else:
             raise PermutoReverseError(f"Cannot assign dict to list index {current_key_or_index}")

        for key, value in reverse_node.items():
            _reconstruct_context_recursive(value, result_json, new_node, key)

    elif isinstance(reverse_node, str): # Should be a JSON Pointer string
        pointer_str = reverse_node
        try:
            resolved_value = jsonpointer.resolve_pointer(result_json, pointer_str)
            if isinstance(parent_context_node, dict):
                parent_context_node[current_key_or_index] = resolved_value
            # No list assignment expected here

        # --- FIX for Error Message ---
        except jsonpointer.JsonPointerException as e:
             # More general message covers syntax errors and lookup failures
             raise PermutoReverseError(
                 f"Error processing pointer '{pointer_str}': {e}"
            ) from e
        except Exception as e:
             raise PermutoReverseError(
                 f"Unexpected error processing pointer '{pointer_str}': {e}"
             ) from e
        # --- End FIX ---

    else:
         type_name = type(reverse_node).__name__
         raise PermutoReverseError(
             f"Invalid node type encountered in reverse template at '{current_key_or_index}'. Expected dict or str (JSON Pointer), found: {type_name}"
         )


# --- Public API Functions ---
# (apply, create_reverse_template, apply_reverse remain unchanged from previous version)
def apply(
    template_json: TemplateType,
    context: ContextType,
    options: Optional[Options] = None
) -> JsonType:
    opts = options if options is not None else Options()
    opts.validate()
    template_copy = copy.deepcopy(template_json)
    active_paths: Set[str] = set()
    return _process_node(template_copy, context, opts, active_paths)


def create_reverse_template(
    original_template: TemplateType,
    options: Optional[Options] = None
) -> Dict[str, Any]:
    opts = options if options is not None else Options()
    opts.validate()
    if opts.enable_string_interpolation:
        raise PermutoReverseError("Cannot create a reverse template when string interpolation is enabled in options.")
    reverse_template: Dict[str, Any] = {}
    initial_pointer_str = ""
    _build_reverse_template_recursive(
        original_template,
        initial_pointer_str,
        reverse_template,
        opts
    )
    return reverse_template


def apply_reverse(
    reverse_template: Dict[str, Any],
    result_json: JsonType
) -> ContextType:
    if not isinstance(reverse_template, dict):
        raise PermutoReverseError("Reverse template root must be a dictionary.")
    reconstructed_context: Dict[str, Any] = {}
    for key, value in reverse_template.items():
        # Use _reconstruct_context_recursive to handle nesting
        _reconstruct_context_recursive(value, result_json, reconstructed_context, key)
    return reconstructed_context
