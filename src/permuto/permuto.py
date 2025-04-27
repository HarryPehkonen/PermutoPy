# src/permuto/permuto.py
import copy
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Union

try:
    import jsonpointer
except ImportError:
    raise ImportError("The 'jsonpointer' library is required for permuto. Please install it: pip install jsonpointer")


from .exceptions import (
    PermutoCycleError,
    PermutoException,
    PermutoInvalidOptionsError,
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
    """
    Configuration options for Permuto processing.

    Controls how template processing behaves, such as variable delimiters,
    handling of missing keys, and string interpolation.

    Attributes:
        variable_start_marker (str): The string marking the beginning of a
            placeholder. Defaults to "${".
        variable_end_marker (str): The string marking the end of a placeholder.
            Defaults to "}".
        on_missing_key (str): Defines behavior when a key or path specified in
            a placeholder is not found in the context. Must be 'ignore' (leave
            the placeholder as is) or 'error' (raise PermutoMissingKeyError).
            Defaults to "ignore".
        enable_string_interpolation (bool): If True, placeholders within larger
            strings (e.g., "Hello, ${user.name}!") are substituted. If False,
            only exact match placeholders (e.g., the entire string is "${user.name}")
            are substituted; others are treated as literals. Reverse operations
            require this to be False. Defaults to False.
    """
    def __init__(
        self,
        variable_start_marker: str = "${",
        variable_end_marker: str = "}",
        on_missing_key: str = "ignore",  # 'ignore' or 'error'
        enable_string_interpolation: bool = False,
    ):
        """
        Initializes the Options object.

        Args:
            variable_start_marker: The starting delimiter for placeholders.
            variable_end_marker: The ending delimiter for placeholders.
            on_missing_key: Behavior for missing keys ('ignore' or 'error').
            enable_string_interpolation: Enable interpolation in non-exact strings.
        """
        self.variable_start_marker = variable_start_marker
        self.variable_end_marker = variable_end_marker
        self.on_missing_key = on_missing_key
        self.enable_string_interpolation = enable_string_interpolation
        self.validate()

    def validate(self) -> None:
        """
        Validates the options.

        Raises:
            PermutoInvalidOptionsError: If any option value is invalid (e.g.,
                empty markers, identical markers, invalid on_missing_key value).
        """
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
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
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
    full_placeholder_for_error: str,
) -> Union[JsonType, object]:
    if not path:
        if options.on_missing_key == "error":
            raise PermutoMissingKeyError("Path cannot be empty within placeholder", path)
        return _NOT_FOUND

    current_obj = context
    segments = path.split(".")
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
                if isinstance(current_obj, list) and potential_key.isdigit() and lookahead == 0:
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
                 path,
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
    active_paths: Set[str],
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
        # Successfully resolved (value could be None)
        if isinstance(resolved_value, str):
            return _process_string(resolved_value, context, options, active_paths)
        return resolved_value # Return actual value (including None)


def _process_string(
    template_str: str,
    context: ContextType,
    options: Options,
    active_paths: Set[str],
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

    # --- Check for Exact Match Placeholder ---
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

    # --- Handle based on Interpolation Option ---
    if not options.enable_string_interpolation:
        return template_str

    # --- Perform Interpolation ---
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
                placeholder_path, full_placeholder, context, options, active_paths,
            )
            result_parts.append(_stringify_json(resolved_value))

        current_pos = end_pos + end_len

    return "".join(result_parts)


def _process_node(
    node: TemplateType,
    context: ContextType,
    options: Options,
    active_paths: Set[str],
) -> JsonType:
    """Recursively processes a node in the template."""
    if isinstance(node, dict):
        result_obj = {}
        for key, val in node.items():
            result_obj[key] = _process_node(val, context, options, active_paths)
        return result_obj
    if isinstance(node, list):
        return [_process_node(element, context, options, active_paths) for element in node]
    if isinstance(node, str):
        return _process_string(node, context, options, active_paths)
    return node


# --- Reverse Template Helpers ---

def _insert_pointer_at_context_path(
    reverse_template_node: Dict[str, Any],
    context_path: str, # Dot notation path for context structure
    pointer_to_insert: str, # JSON Pointer string for result location
) -> None:
    """
    Inserts the result pointer into the nested reverse template structure,
    using simple dot splitting to define the structure.
    """
    if not context_path:
         raise PermutoReverseError("[Internal] Invalid empty context path.")

    # Basic path validation for empty segments
    if context_path.startswith(".") or context_path.endswith(".") or ".." in context_path:
        raise PermutoReverseError(f"[Internal] Invalid context path format: {context_path}")

    segments = context_path.split(".")
    current_node = reverse_template_node

    for i, segment in enumerate(segments):
        if not segment: # Check again after split
            raise PermutoReverseError(f"[Internal] Invalid context path format (empty segment): {context_path}")

        is_last_segment = (i == len(segments) - 1)

        if is_last_segment:
            # Assign the pointer string at the final key (segment), overwriting if needed.
            # If the key already exists and is a dict, this overwrites it.
            # If the key exists and is not a dict, it also overwrites.
            # Consider if we need conflict detection (e.g., "${var}" vs "${var.sub}")
            # Current behavior: Last processed placeholder wins for a given segment structure.
            # If a dict exists where the pointer should go, it gets replaced by the pointer.
            # If a pointer exists where a dict should go (from a deeper path), the dict replaces it.
            existing_val = current_node.get(segment)
            if isinstance(existing_val, dict):
                 # Overwriting an existing structure with a primitive pointer
                 # This is necessary for cases like {"a": "${var.sub}", "b": "${var}"} -> final state {"var": "/b"}
                 pass # Allow overwrite below
            current_node[segment] = pointer_to_insert
        else:
            # Need to descend. Get or create the next node.
            existing_val = current_node.get(segment)
            if not isinstance(existing_val, dict):
                # If it doesn't exist, or is not a dict (it's a pointer string),
                # overwrite/create a dict. This handles cases where a primitive path
                # like "${var}" was processed before a nested path like "${var.sub}".
                current_node[segment] = {}
            # Descend into the object node.
            current_node = current_node[segment]


def _build_reverse_template_recursive(
    current_template_node: TemplateType,
    current_result_pointer_str: str, # JSON Pointer string for result location
    reverse_template_ref: Dict[str, Any], # The root reverse template being built
    options: Options,
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
            # Validate content doesn't contain markers
            if start_marker not in content and end_marker not in content:
                context_path = content
                # Validate the extracted context_path format *before* declaring it an exact match
                if context_path and not (context_path.startswith(".") or context_path.endswith(".") or ".." in context_path):
                    # Only if the path is non-empty AND has a valid format
                    is_exact_match = True

        if is_exact_match:
            try:
                _insert_pointer_at_context_path(
                    reverse_template_ref, context_path, current_result_pointer_str,
                )
            except PermutoReverseError as e:
                 raise PermutoReverseError(f"Error building reverse template for context path '{context_path}': {e}") from e
            except Exception as e:
                 raise PermutoReverseError(f"Unexpected error inserting pointer for context path '{context_path}': {e}") from e

def _reconstruct_context_recursive(
    reverse_node: JsonType,
    result_json: JsonType,
    parent_context_node: Union[Dict, List], # The node being built (dict or list)
    current_key_or_index: Union[str, int], # Key/Index in the parent to assign to
) -> None:
    """Recursively reconstructs the context using the reverse template."""

    if isinstance(reverse_node, dict):
        new_node: Dict[str, Any] = {}
        if isinstance(parent_context_node, dict):
            parent_context_node[current_key_or_index] = new_node
        # NOTE: Assigning dict to list index is intentionally not supported here,
        # as reverse templates are expected to mirror context structure (primarily dicts).
        # elif isinstance(parent_context_node, list):
        #     # This case would require careful handling if context could be list root
        #     pass
        else:
             raise PermutoReverseError(f"Cannot assign dict to non-dict parent node at '{current_key_or_index}'")

        for key, value in reverse_node.items():
            _reconstruct_context_recursive(value, result_json, new_node, key)

    elif isinstance(reverse_node, str): # Should be a JSON Pointer string
        pointer_str = reverse_node
        try:
            resolved_value = jsonpointer.resolve_pointer(result_json, pointer_str)
            if isinstance(parent_context_node, dict):
                parent_context_node[current_key_or_index] = resolved_value
            # NOTE: Assigning resolved value to list index intentionally not supported.
            # elif isinstance(parent_context_node, list):
            #     # This case would require careful handling.
            #     pass
            else:
                raise PermutoReverseError(f"Cannot assign value to non-dict parent node at '{current_key_or_index}'")

        except jsonpointer.JsonPointerException as e:
             # Covers syntax errors and lookup failures (path not found)
             raise PermutoReverseError(
                 f"Error processing pointer '{pointer_str}': {e}. Ensure the pointer is valid and exists in the result JSON.",
            ) from e
        except Exception as e:
             raise PermutoReverseError(
                 f"Unexpected error processing pointer '{pointer_str}': {e}",
             ) from e

    else:
         type_name = type(reverse_node).__name__
         raise PermutoReverseError(
             f"Invalid node type encountered in reverse template structure at/under '{current_key_or_index}'. Expected dict or str (JSON Pointer), found: {type_name}",
         )


# --- Public API Functions ---

def apply(
    template_json: TemplateType,
    context: ContextType,
    options: Optional[Options] = None,
) -> JsonType:
    """
    Applies context data to a JSON template structure.

    Recursively traverses the template. When a string node is encountered,
    it checks for placeholders defined by `options.variable_start_marker` and
    `options.variable_end_marker`.

    - If a string is an *exact match* for a placeholder (e.g., "${user.name}"),
      it's replaced by the corresponding value from the context, preserving the
      original data type (number, boolean, null, list, object, string).
      If the resolved value is itself a string containing placeholders, it's
      recursively processed using the same `options`.
    - If `options.enable_string_interpolation` is True and a string contains
      placeholders but isn't an exact match (e.g., "Hello, ${user.name}!"),
      the placeholders are substituted with their string representations
      (non-strings are typically converted to compact JSON strings).
    - If `options.enable_string_interpolation` is False (default), strings
      that are not exact matches are treated as literals.

    Handles nested data access using dot notation (e.g., "${user.profile.email}").
    Detects and prevents cyclical substitutions.

    Note: The input template is deep-copied before processing to avoid mutation.

    Args:
        template_json: A JSON-like structure (dict, list, primitive) serving
                       as the template.
        context: A dictionary containing the data to substitute into the template.
        options: An optional Options object to customize behavior. If None,
                 default options are used.

    Returns:
        A new JSON-like structure representing the template with substitutions applied.

    Raises:
        PermutoInvalidOptionsError: If the provided Options are invalid.
        PermutoCycleError: If a cyclical dependency is detected during substitution.
        PermutoMissingKeyError: If `options.on_missing_key` is 'error' and a
                                required key/path is not found in the context.
        PermutoException: For other potential processing errors.
        TypeError: If the input template or context is not of the expected type.
    """
    opts = options if options is not None else Options()
    opts.validate()
    # Perform a deep copy to avoid modifying the original template
    try:
        template_copy = copy.deepcopy(template_json)
    except Exception as e:
        raise TypeError(f"Failed to deepcopy the template. Ensure it's copyable: {e}") from e

    if not isinstance(context, dict):
        raise TypeError("Context must be a dictionary.")

    active_paths: Set[str] = set()
    return _process_node(template_copy, context, opts, active_paths)


def create_reverse_template(
    original_template: TemplateType,
    options: Optional[Options] = None,
) -> Dict[str, Any]:
    """
    Creates a reverse template mapping context paths to result JSON Pointers.

    Scans the `original_template` for *exact match* placeholders (e.g., "${key.path}").
    For each found placeholder, it determines its location in the template structure
    (represented as a JSON Pointer string, e.g., "/user/name") and the corresponding
    context variable path (e.g., "key.path").

    It builds a dictionary structure mirroring the *context*, where the leaf values
    are the JSON Pointer strings pointing to where that context variable's value
    is expected to be found in the *result* JSON generated by `apply()`.

    This function requires `options.enable_string_interpolation` to be False,
    as interpolation makes reversing ambiguous.

    Args:
        original_template: The JSON-like template structure that was used (or
                           will be used) with `apply()`.
        options: An optional Options object. Must have
                 `enable_string_interpolation=False`. Delimiters from options
                 are used to find placeholders. If None, default options are used.

    Returns:
        A dictionary representing the reverse template structure.

    Raises:
        PermutoInvalidOptionsError: If the provided Options are invalid.
        PermutoReverseError: If `options.enable_string_interpolation` is True,
                             or if errors occur during reverse template construction
                             (e.g., invalid context path formats).
        TypeError: If the input template is not of the expected type.
    """
    opts = options if options is not None else Options()
    opts.validate()
    if opts.enable_string_interpolation:
        raise PermutoReverseError("Cannot create a reverse template when string interpolation is enabled in options.")

    # Ensure template is a valid structure (basic check)
    if not isinstance(original_template, (dict, list, str, int, float, bool, type(None))):
         raise TypeError("Original template must be a valid JSON-like structure.")

    reverse_template: Dict[str, Any] = {}
    initial_pointer_str = "" # Root pointer
    _build_reverse_template_recursive(
        original_template,
        initial_pointer_str,
        reverse_template,
        opts,
    )
    return reverse_template


def apply_reverse(
    reverse_template: Dict[str, Any],
    result_json: JsonType,
) -> ContextType:
    """
    Reconstructs the original context data from a result using a reverse template.

    Traverses the `reverse_template` structure. Each leaf node in the
    `reverse_template` is expected to be a string containing a JSON Pointer
    (e.g., "/user/name").

    For each leaf node (JSON Pointer), this function looks up the value at that
    pointer location within the `result_json`. It then builds a new dictionary
    structure, placing the extracted values at the paths defined by the
    `reverse_template` structure itself.

    Example:
        If `reverse_template` is `{"user": {"name": "/output_name"}}`
        and `result_json` is `{"output_name": "Alice"}`,
        this function will return `{"user": {"name": "Alice"}}`.

    Args:
        reverse_template: The reverse template dictionary generated by
                          `create_reverse_template()`.
        result_json: The JSON-like structure that was produced by applying the
                     original template (using `apply()` with interpolation disabled).

    Returns:
        A dictionary representing the reconstructed context data.

    Raises:
        PermutoReverseError: If the `reverse_template` is malformed (e.g., root
                             is not a dict, leaf nodes are not strings, JSON
                             pointers are invalid or do not resolve in the
                             `result_json`).
        TypeError: If inputs are not of the expected types.
        jsonpointer.JsonPointerException: If a pointer in the reverse template
                                          cannot be resolved in the result_json.
    """
    if not isinstance(reverse_template, dict):
        raise PermutoReverseError("Reverse template root must be a dictionary.")
    # Basic check for result_json type might be useful too
    if not isinstance(result_json, (dict, list, str, int, float, bool, type(None))):
         raise TypeError("Result JSON must be a valid JSON-like structure.")

    reconstructed_context: Dict[str, Any] = {}
    # Iterate through the top level of the reverse template
    for key, value in reverse_template.items():
        # Use the recursive helper to handle nested structures and assignments
        _reconstruct_context_recursive(value, result_json, reconstructed_context, key)

    return reconstructed_context
