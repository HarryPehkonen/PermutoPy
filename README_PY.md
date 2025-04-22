** Installation and Usage:**

pip install jsonpointer pytest

1.  **Install:**
    ```bash
    # Navigate to the permuto-py directory (containing pyproject.toml)
    pip install .
    # Or for development (editable install):
    pip install -e .
    # To include test dependencies:
    pip install -e ".[test]"
    ```
2.  **Use in Python:**
    ```python
    import permuto
    import json

    template = json.loads("""
        {
            "user_info": {
                "name": "${user.name}",
                "id_str": "User ID: ${user.id}"
            },
            "enabled": "${config.active}"
        }
    """)

    context = {
        "user": {"name": "Bob", "id": 456},
        "config": {"active": True}
    }

    try:
        # Interpolation Disabled (Default)
        opts_off = permuto.Options()
        result_off = permuto.apply(template, context, opts_off)
        print("--- Result (Interpolation OFF) ---")
        print(json.dumps(result_off, indent=4))
        # Expected: name is Bob, id_str is literal, enabled is True

        # Interpolation Enabled
        opts_on = permuto.Options(enable_string_interpolation=True)
        result_on = permuto.apply(template, context, opts_on)
        print("\n--- Result (Interpolation ON) ---")
        print(json.dumps(result_on, indent=4))
        # Expected: name is Bob, id_str is "User ID: 456", enabled is True

        # Reverse Operation (using result_off)
        if not opts_off.enable_string_interpolation:
             print("\n--- Reverse Operation ---")
             rev_template = permuto.create_reverse_template(template, opts_off)
             print("Reverse Template:")
             print(json.dumps(rev_template, indent=4))
             # Expected: {"user": {"name": "/user_info/name"}, "config": {"active": "/enabled"}}

             reconstructed = permuto.apply_reverse(rev_template, result_off)
             print("\nReconstructed Context:")
             print(json.dumps(reconstructed, indent=4))
             # Expected: {"user": {"name": "Bob"}, "config": {"active": True}}


    except permuto.PermutoException as e:
        print(f"Permuto Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

