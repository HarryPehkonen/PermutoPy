# src/permuto/cli.py
import argparse
import json
import sys
import os # Added for better error messages

# Import necessary components from the permuto library
try:
    # Adjust import path based on how the CLI is run or installed
    # If run directly using `python src/permuto/cli.py`, this works
    # If installed and run as a script, `from . import ...` might be needed,
    # but relative imports are tricky in top-level scripts.
    # Using `permuto.` prefix assumes the package structure is accessible.
    from permuto import apply, Options
    from permuto.exceptions import (
        PermutoException,
        PermutoInvalidOptionsError,
        PermutoCycleError,
        PermutoMissingKeyError
    )
except ImportError:
    # Fallback if running script from repository root (e.g., python src/permuto/cli.py)
    # Or handle cases where the package isn't installed properly in the environment
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from permuto import apply, Options
        from permuto.exceptions import (
            PermutoException,
            PermutoInvalidOptionsError,
            PermutoCycleError,
            PermutoMissingKeyError
        )
    except ImportError as e:
        print(f"Error: Could not import the permuto library. "
              f"Ensure it's installed or your PYTHONPATH is correct. Details: {e}",
              file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Apply a Permuto template to a context JSON file.",
        epilog="Outputs the resulting JSON to standard output."
    )

    # --- Arguments ---
    parser.add_argument(
        "template_file",
        metavar="<template_file>",
        type=str,
        help="Path to the input JSON template file."
    )
    parser.add_argument(
        "context_file",
        metavar="<context_file>",
        type=str,
        help="Path to the input JSON data context file."
    )

    # --- Options ---
    parser.add_argument(
        "--on-missing-key",
        choices=["ignore", "error"],
        default="ignore",
        help="Behavior for missing keys ('ignore' or 'error'). Default: ignore."
    )
    parser.add_argument(
        "--string-interpolation",
        action="store_true", # Flag: presence enables interpolation
        default=False,      # Default is interpolation OFF
        help="Enable string interpolation for non-exact matches. "
             "If omitted, non-exact match strings are treated as literals."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="${",
        help="Set the variable start delimiter. Default: ${"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="}",
        help="Set the variable end delimiter. Default: }"
    )

    args = parser.parse_args()

    # --- Load Input Files ---
    try:
        with open(args.template_file, 'r') as f:
            template_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Template file not found: {args.template_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in template file '{args.template_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading template file '{args.template_file}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.context_file, 'r') as f:
            context_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Context file not found: {args.context_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in context file '{args.context_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading context file '{args.context_file}': {e}", file=sys.stderr)
        sys.exit(1)


    # --- Configure Options ---
    try:
        options = Options(
            variable_start_marker=args.start,
            variable_end_marker=args.end,
            on_missing_key=args.on_missing_key,
            enable_string_interpolation=args.string_interpolation
        )
        options.validate() # Validate options before applying
    except PermutoInvalidOptionsError as e:
         print(f"Error: Invalid options: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e: # Catch potential other issues creating Options
         print(f"Error configuring options: {e}", file=sys.stderr)
         sys.exit(1)

    # --- Apply Template ---
    try:
        result_json = apply(template_json, context_json, options)
    except PermutoCycleError as e:
        print(f"Error: Cycle detected during substitution. {e}", file=sys.stderr)
        sys.exit(1)
    except PermutoMissingKeyError as e:
        print(f"Error: Missing key/path encountered. {e}", file=sys.stderr)
        sys.exit(1)
    except PermutoException as e: # Catch other specific permuto errors
        print(f"Permuto Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch unexpected errors during apply
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        # Optionally add traceback here for debugging
        # import traceback
        # traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Output Result ---
    try:
        print(json.dumps(result_json, indent=4)) # Pretty print
    except Exception as e:
        print(f"Error serializing result to JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
