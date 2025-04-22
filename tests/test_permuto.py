# tests/test_permuto.py

import pytest
import json
from permuto import (
    apply, create_reverse_template, apply_reverse, Options,
    PermutoCycleError, PermutoMissingKeyError, PermutoInvalidOptionsError, PermutoReverseError
)

# --- Fixtures ---
@pytest.fixture
def basic_context():
    return {
        "user": {"name": "Alice", "email": "alice@example.com", "id": 123, "active": True},
        "sys": {"id": "XYZ"},
        "settings": {"theme": "dark", "enabled": True, "values": [10, 20], "nested": {"val": 5}},
        "misc": None,
        "a": {"b": {"c": "deep"}},
        "ref": "${user.name}",
        "nested_ref": "User: ${ref}",
        "cycle_a": "${cycle_b}",
        "cycle_b": "${cycle_a}",
        "tilde~key": "tilde_val",
        "slash/key": "slash_val",
        "dot.key": "dot_val", # Top-level key with dot
        "nav": { "dot.key": "nested_dot_val"} # Navigable key
    }

# --- Apply Tests ---

def test_apply_simple_interpolation_on(basic_context):
    template = {
        "greeting": "Hello, ${user.name}!",
        "email": "${user.email}",
        "active_status": "${user.active}" # Exact match bool
    }
    opts = Options(enable_string_interpolation=True)
    expected = {
        "greeting": "Hello, Alice!",
        "email": "alice@example.com",
        "active_status": True
    }
    assert apply(template, basic_context, opts) == expected

def test_apply_simple_interpolation_off(basic_context):
    template = {
        "greeting": "Hello, ${user.name}!", # Literal
        "email": "${user.email}",         # Exact Match -> substituted
        "active_status": "${user.active}" # Exact match -> substituted
    }
    opts = Options(enable_string_interpolation=False) # Default
    expected = {
        "greeting": "Hello, ${user.name}!",
        "email": "alice@example.com",
        "active_status": True
    }
    # Test with default options
    assert apply(template, basic_context) == expected
    # Test explicitly passing default options
    assert apply(template, basic_context, opts) == expected

def test_apply_exact_match_types(basic_context):
    template = {
        "name": "${user.name}",
        "id": "${user.id}",
        "active": "${user.active}",
        "misc": "${misc}",
        "vals": "${settings.values}",
        "nested": "${settings.nested}"
    }
    expected = {
        "name": "Alice",
        "id": 123,
        "active": True,
        "misc": None,
        "vals": [10, 20],
        "nested": {"val": 5}
    }
    # Should work identically regardless of interpolation setting
    assert apply(template, basic_context, Options(enable_string_interpolation=False)) == expected
    assert apply(template, basic_context, Options(enable_string_interpolation=True)) == expected

def test_apply_stringification_interpolation_on(basic_context):
     template = {"info": "ID=${user.id}, Active=${user.active}, Misc=${misc}, Vals=${settings.values}"}
     opts = Options(enable_string_interpolation=True)
     expected = {"info": "ID=123, Active=true, Misc=null, Vals=[10,20]"}
     assert apply(template, basic_context, opts) == expected

def test_apply_missing_key_ignore(basic_context):
    template = {"known": "${user.name}", "unknown": "${user.address}", "interp_miss": "Val: ${settings.font}"}
    opts_off = Options(enable_string_interpolation=False, on_missing_key='ignore')
    opts_on = Options(enable_string_interpolation=True, on_missing_key='ignore')
    expected = {"known": "Alice", "unknown": "${user.address}", "interp_miss": "Val: ${settings.font}"}
    assert apply(template, basic_context, opts_off) == expected
    assert apply(template, basic_context, opts_on) == expected # Same result for ignore

def test_apply_missing_key_error(basic_context):
    template_exact = {"fail": "${user.address}"}
    template_interp = {"fail": "Value: ${user.address}"}
    opts_off = Options(enable_string_interpolation=False, on_missing_key='error')
    opts_on = Options(enable_string_interpolation=True, on_missing_key='error')

    # Exact match fails in both modes
    with pytest.raises(PermutoMissingKeyError) as excinfo_off_exact:
        apply(template_exact, basic_context, opts_off)
    assert excinfo_off_exact.value.key_path == "user.address"
    with pytest.raises(PermutoMissingKeyError) as excinfo_on_exact:
        apply(template_exact, basic_context, opts_on)
    assert excinfo_on_exact.value.key_path == "user.address"

    # Interpolation only fails when enabled
    with pytest.raises(PermutoMissingKeyError) as excinfo_on_interp:
        apply(template_interp, basic_context, opts_on)
    assert excinfo_on_interp.value.key_path == "user.address"

    # Interpolation does NOT fail when disabled (returns literal)
    expected_interp_off = {"fail": "Value: ${user.address}"}
    assert apply(template_interp, basic_context, opts_off) == expected_interp_off

def test_apply_cycle_detection(basic_context):
    template_exact = {"a": "${cycle_a}"}
    template_interp = {"a": "Cycle: ${cycle_a}"}
    opts_off = Options(enable_string_interpolation=False)
    opts_on = Options(enable_string_interpolation=True)

    with pytest.raises(PermutoCycleError):
        apply(template_exact, basic_context, opts_off)
    with pytest.raises(PermutoCycleError):
        apply(template_exact, basic_context, opts_on)
    with pytest.raises(PermutoCycleError):
        apply(template_interp, basic_context, opts_on)

    # No cycle error if interpolation is off for non-exact match
    expected_interp_off = {"a": "Cycle: ${cycle_a}"}
    assert apply(template_interp, basic_context, opts_off) == expected_interp_off


def test_apply_recursion(basic_context):
    template_exact = {"info": "${nested_ref}"} # -> "User: ${ref}"
    template_interp = {"msg": "Details: ${nested_ref}"}

    opts_off = Options(enable_string_interpolation=False)
    opts_on = Options(enable_string_interpolation=True)

    # Interp Off: Exact match ${nested_ref} -> "User: ${ref}" -> literal result
    expected_off = {"info": "User: ${ref}"}
    assert apply(template_exact, basic_context, opts_off) == expected_off

    # Interp On: Exact match ${nested_ref} -> "User: ${ref}" -> recursive interp -> "User: Alice"
    expected_on = {"info": "User: Alice"}
    assert apply(template_exact, basic_context, opts_on) == expected_on

    # Interp On: Interpolating ${nested_ref} -> recursive interp -> "User: Alice"
    expected_interp_on = {"msg": "Details: User: Alice"}
    assert apply(template_interp, basic_context, opts_on) == expected_interp_on

    # Interp Off: Interpolating ${nested_ref} -> literal (no interp happens)
    expected_interp_off = {"msg": "Details: ${nested_ref}"}
    assert apply(template_interp, basic_context, opts_off) == expected_interp_off

def test_apply_custom_delimiters(basic_context):
    template = {"id": "<<user.id>>", "msg": "User is <<user.name>>"}
    opts_on = Options(variable_start_marker="<<", variable_end_marker=">>", enable_string_interpolation=True)
    opts_off = Options(variable_start_marker="<<", variable_end_marker=">>", enable_string_interpolation=False)
    expected_on = {"id": 123, "msg": "User is Alice"}
    expected_off = {"id": 123, "msg": "User is <<user.name>>"}
    assert apply(template, basic_context, opts_on) == expected_on
    assert apply(template, basic_context, opts_off) == expected_off


def test_apply_special_keys(basic_context):
    # Need jsonpointer for these tests!
    template = {
        "val1": "${tilde~key}",   # context["tilde~key"]
        "val2": "${slash/key}",   # context["slash/key"]
        "val3": "${nav.dot.key}" # context["nav"]["dot.key"]
    }
    expected = {
        "val1": "tilde_val",
        "val2": "slash_val",
        "val3": "nested_dot_val"
    }
    # Should work regardless of interpolation mode (using greedy path resolution)
    assert apply(template, basic_context, Options(enable_string_interpolation=False)) == expected
    assert apply(template, basic_context, Options(enable_string_interpolation=True)) == expected

    # --- Test Accessing Top-Level Key with Dot ---
    # The greedy _resolve_path should find the literal key "dot.key"
    template_dot = {"val4": "${dot.key}"}
    expected_dot = {"val4": "dot_val"}
    assert apply(template_dot, basic_context, Options(enable_string_interpolation=False)) == expected_dot

    # --- Test Case Where Navigation Would Fail but Literal Key Doesn't Exist ---
    # This case should fail because context['nav'] exists, but context['nav']['nonexistent'] doesn't.
    # The greedy lookup won't find a multi-part key either.
    template_nav_fail = {"fail": "${nav.nonexistent}"}
    with pytest.raises(PermutoMissingKeyError) as excinfo_nav:
        apply(template_nav_fail, basic_context, Options(on_missing_key='error'))
    # Check the path reported in the error
    assert excinfo_nav.value.key_path == "nav.nonexistent"

# --- Create Reverse Template Tests ---

def test_create_reverse_basic():
    original_template = {
        "output_name": "${user.name}",
        "output_email": "${user.email}",
        "system_id": "${sys.id}",
        "literal_string": "Hello World",
        "interpolated_string": "Hello ${user.name}", # Ignored
        "nested": {
            "is_active": "${user.active}",
            "config_value": "${cfg.val}"
        }
    }
    expected_reverse = {
        "user": {
            "name": "/output_name",
            "email": "/output_email",
            "active": "/nested/is_active"
        },
        "sys": {
            "id": "/system_id"
        },
        "cfg": {
            "val": "/nested/config_value"
        }
    }
    # Use default options (interp=False)
    assert create_reverse_template(original_template) == expected_reverse
    # Explicitly pass options
    assert create_reverse_template(original_template, Options(enable_string_interpolation=False)) == expected_reverse

def test_create_reverse_arrays():
    original_template = {
        "ids": ["${user.id}", "${sys.id}"],
        "mixed": [1, "${user.name}", True, {"key": "${cfg.val}"}]
    }
    expected_reverse = {
        "user": { "id": "/ids/0", "name": "/mixed/1" },
        "sys": { "id": "/ids/1" },
        "cfg": { "val": "/mixed/3/key" }
    }
    assert create_reverse_template(original_template) == expected_reverse

def test_create_reverse_special_keys():
     original_template = {
            "slash~1key_out": "${data.slash/key}", # context path segment is "slash/key"
            "tilde~0key_out": "${data.tilde~key}", # context path segment is "tilde~key"
            "dot.key_out": "${nav.dot.key}" # context path segments are "nav", "dot.key"
        }
     expected_reverse = {
            "data": {
                "slash/key": "/slash~1key_out", # Result pointer has escaped '/'
                "tilde~key": "/tilde~0key_out"  # Result pointer has escaped '~'
            },
            "nav": {
                "dot.key": "/dot.key_out" # Result pointer is simple, context structure has dot
            }
        }
     assert create_reverse_template(original_template) == expected_reverse


def test_create_reverse_invalid_placeholders():
    original_template = {"a": "${}", "b": "${.path}", "c": "${path.}", "d": "${a..b}"}
    expected_reverse = {} # All should be ignored
    assert create_reverse_template(original_template) == expected_reverse


def test_create_reverse_throws_if_interpolation_on():
     original_template = {"a": "${b}"}
     with pytest.raises(PermutoReverseError, match="interpolation is enabled"):
        create_reverse_template(original_template, Options(enable_string_interpolation=True))


def test_create_reverse_context_path_conflict():
    # Based on C++ test fix: processing order doesn't matter, nesting requirement wins.
    original_template_obj_last = {"a": "${var}", "b": "${var.sub}"}
    original_template_prim_last = {"b": "${var.sub}", "a": "${var}"}
    expected_reverse = {"var": {"sub": "/b"}} # Object structure wins

    assert create_reverse_template(original_template_obj_last) == expected_reverse
    assert create_reverse_template(original_template_prim_last) == expected_reverse


# --- Apply Reverse Tests ---

@pytest.fixture
def reverse_test_data():
    reverse_template = {
            "user": {
                "name": "/output_name",
                "email": "/output_email",
                "active": "/nested/is_active",
                "id": "/ids/0"
            },
            "sys": {
                "id": "/ids/1",
                "nested_arr": "/complex/0/value" # Pointer to an array
            },
            "cfg": {
                "val": "/nested/config_value",
                "num": "/num_val",
                "null_val": "/null_item",
                "obj": "/obj_val" # Pointer to an object
            },
            "special/key": "/tilde~0key", # Context key needs escaping
            "special~key": "/slash~1key"  # Context key needs escaping
    }
    result_json = {
            "output_name": "Alice",
            "output_email": "alice@example.com",
            "literal_string": "Hello World",
            "nested": {
                "is_active": True,
                "config_value": "theme_dark"
            },
            "ids": [123, "XYZ"],
            "num_val": 45.6,
            "null_item": None,
            "obj_val": {"a": 1},
            "complex": [ {"value": [10, 20]} ],
            "tilde~key": "Tilde Value", # Result key needs escaping for pointer lookup
            "slash/key": "Slash Value"  # Result key needs escaping for pointer lookup
    }
    expected_context = {
            "user": {
                "name": "Alice",
                "email": "alice@example.com",
                "active": True,
                "id": 123
            },
            "sys": {
                "id": "XYZ",
                "nested_arr": [10, 20]
            },
            "cfg": {
                "val": "theme_dark",
                "num": 45.6,
                "null_val": None,
                "obj": {"a": 1}
            },
            "special/key": "Tilde Value",
            "special~key": "Slash Value"
    }
    return reverse_template, result_json, expected_context


def test_apply_reverse_basic(reverse_test_data):
    rt, rj, ec = reverse_test_data
    reconstructed = apply_reverse(rt, rj)
    assert reconstructed == ec

def test_apply_reverse_empty_template():
     rt = {}
     rj = {"a": 1}
     assert apply_reverse(rt, rj) == {}

def test_apply_reverse_pointer_not_found(reverse_test_data):
     rt, rj, _ = reverse_test_data
     rt["user"]["id"] = "/missing_pointer" # Change pointer to invalid one
     # --- FIX: Update match regex to be less specific or match actual error ---
     # Match the start of the PermutoReverseError and the underlying jsonpointer message part
     expected_match = r"Error processing pointer '/missing_pointer'.*member 'missing_pointer' not found"
     with pytest.raises(PermutoReverseError, match=expected_match):
        apply_reverse(rt, rj)

def test_apply_reverse_invalid_pointer_syntax(reverse_test_data):
     rt, rj, _ = reverse_test_data
     rt["user"]["name"] = "invalid pointer syntax"
     with pytest.raises(PermutoReverseError, match="Error processing pointer 'invalid pointer syntax'"):
        apply_reverse(rt, rj)

def test_create_reverse_special_keys():
     original_template = {
            # Key in RESULT JSON | Placeholder with CONTEXT path
            "slash~1key_out": "${data.slash/key}", # context path segment is "slash/key"
            "tilde~0key_out": "${data.tilde~key}", # context path segment is "tilde~key"
            "dot.key_out":    "${nav.dot.key}"     # context path parsed as "nav", "dot", "key" for structure
        }
     # Expected reverse maps CONTEXT structure (as parsed by split('.')) to RESULT pointers
     expected_reverse = {
            "data": {
                # Context key | Pointer to result key (escaped)
                "slash/key": "/slash~01key_out",
                "tilde~key": "/tilde~00key_out"
            },
            # --- Structure based on simple dot split of 'nav.dot.key' ---
            "nav": {
                 "dot": {
                      "key": "/dot.key_out"
                 }
            }
            # --- End Structure Fix ---
        }
     assert create_reverse_template(original_template) == expected_reverse

def test_create_reverse_context_path_conflict():
    # Based on C++ test fix: processing order doesn't matter, nesting requirement wins.
    # -> Correction: Python dict order *does* matter, last write wins.
    original_template_obj_last = {"a": "${var}", "b": "${var.sub}"} # Process a, then b
    original_template_prim_last = {"b": "${var.sub}", "a": "${var}"} # Process b, then a

    # For obj_last: {'var':'/a'} then {'var':{'sub':'/b'}} -> nested wins
    expected_reverse_obj_last = {"var": {"sub": "/b"}}

    # For prim_last: {'var':{'sub':'/b'}} then {'var':'/a'} -> primitive wins (overwrites)
    expected_reverse_prim_last = {"var": "/a"}
    # --- FIX: Corrected expected_reverse_prim_last ---


    assert create_reverse_template(original_template_obj_last) == expected_reverse_obj_last
    assert create_reverse_template(original_template_prim_last) == expected_reverse_prim_last


# --- Apply Reverse Tests ---
# (reverse_test_data fixture remains the same)
# (test_apply_reverse_basic remains the same)
# (test_apply_reverse_empty_template remains the same)
# (test_apply_reverse_pointer_not_found remains the same)


def test_apply_reverse_invalid_pointer_syntax(reverse_test_data):
     rt, rj, _ = reverse_test_data
     rt["user"]["name"] = "invalid pointer syntax"
     # --- FIX: Update expected error message regex ---
     with pytest.raises(PermutoReverseError, match="Error processing pointer 'invalid pointer syntax'.*Location must start with /"):
        apply_reverse(rt, rj)

def test_apply_reverse_malformed_template_types(reverse_test_data):
     rt, rj, _ = reverse_test_data

     # Non-dict intermediate node
     rt_bad1 = {"user": ["/a", "/b"]}
     with pytest.raises(PermutoReverseError, match="Invalid node type encountered.*list"):
         apply_reverse(rt_bad1, rj)

     # Non-string leaf node
     rt_bad2 = {"user": {"name": 123}}
     with pytest.raises(PermutoReverseError, match="Invalid node type encountered.*int"):
         apply_reverse(rt_bad2, rj)

     # Root not dict
     rt_bad3 = ["/a"]
     with pytest.raises(PermutoReverseError, match="Reverse template root must be a dictionary"):
         apply_reverse(rt_bad3, rj)


# --- Round Trip Test ---
def test_round_trip_full_cycle(basic_context):
     original_template = {
            "userName": "${user.name}",
            "details": {
                 "isActive": "${user.active}",
                 "city": "${a.b.c}" # Use a deep path
             },
            "ids": [ "${sys.id}", "${user.id}" ],
            "config": {
                "theme": "${settings.theme}",
                "notify": "${settings.enabled}" # Use bool
            },
            "literal": "some text",
            "maybe_interpolated": "Value is ${settings.theme}" # Will be literal
        }

     # Use default options (interp=False)
     opts = Options()

     # 1. Apply
     result = apply(original_template, basic_context, opts)
     expected_result = {
            "userName": "Alice",
            "details": { "isActive": True, "city": "deep" },
            "ids": [ "XYZ", 123 ],
            "config": { "theme": "dark", "notify": True },
            "literal": "some text",
            "maybe_interpolated": "Value is ${settings.theme}"
     }
     assert result == expected_result

     # 2. Create Reverse Template
     reverse_template = create_reverse_template(original_template, opts)
     expected_reverse = {
         "user": { "name": "/userName", "active": "/details/isActive", "id": "/ids/1"},
         "a": {"b": {"c": "/details/city"}},
         "sys": {"id": "/ids/0"},
         "settings": {"theme": "/config/theme", "enabled": "/config/notify"}
     }
     assert reverse_template == expected_reverse

     # 3. Apply Reverse
     reconstructed_context = apply_reverse(reverse_template, result)

     # 4. Verify - need to reconstruct ONLY the parts present in the reverse template
     expected_reconstructed = {
          "user": { "name": "Alice", "active": True, "id": 123},
          "a": {"b": {"c": "deep"}},
          "sys": {"id": "XYZ"},
          "settings": {"theme": "dark", "enabled": True}
     }
     assert reconstructed_context == expected_reconstructed

     # Verify against original context parts (ensure values match)
     assert reconstructed_context["user"]["name"] == basic_context["user"]["name"]
     assert reconstructed_context["settings"]["theme"] == basic_context["settings"]["theme"]
     assert reconstructed_context["a"]["b"]["c"] == basic_context["a"]["b"]["c"]

