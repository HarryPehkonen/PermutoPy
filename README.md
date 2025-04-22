# Permuto

Permuto is a lightweight, flexible templating system designed for generating structured data formats (primarily JSON) by substituting variables from a data context into a template. Its core function, typically named `apply()`, uses a valid JSON document as a template and a JSON-like data structure (object/dictionary/map) as the context.

It excels at dynamically creating JSON payloads, such as those required for various API interactions (e.g., Large Language Model APIs), by providing a focused templating mechanism. Permuto is *not* intended to be a full programming language but focuses on structured data generation based on input data.

A key feature is its support for a **reverse operation** using functions typically named `create_reverse_template()` and `apply_reverse()`. This allows extracting the original data context from a result JSON, provided the result was generated using the corresponding template with *string interpolation disabled*.

Implementations of Permuto exist or are planned for various programming languages.  Please check GitHub for Permuto.

## Features

*   **Valid JSON Templates:** Uses standard, parseable JSON documents as the template input format.
*   **Variable Substitution (`apply` operation):** Replaces placeholders in a template using values from a context.
    *   **Exact Match:** Replaces placeholders that constitute the *entire* string value (e.g., `"${variable}"`) with values from a provided data context. This works regardless of the string interpolation setting.
        *   **Nested Data Access (Dot Notation):** Access values deep within the data context using dot notation (e.g., `${user.profile.email}`). Array indexing via dot notation is generally not supported. Implementations should handle keys containing literal dots (`.`), tildes (`~`), or slashes (`/`) according to standard JSON Pointer escaping logic when converting dot notation.
        *   **Automatic Type Handling:** Intelligently handles data types. When a placeholder like `"${variable}"` or `"${path.to.variable}"` is the *entire* string value in the template, and the corresponding data is a number, boolean, array, object, or null, the placeholder is replaced by the actual value, preserving the correct JSON type. String data results in a standard JSON string output.
    *   **Optional String Interpolation:** Control whether placeholders within larger strings are processed via a configuration option (e.g., `enableStringInterpolation`).
        *   **Enabled:** Substitutes variables within larger strings (e.g., `"Hello, ${user.name}!"`). Non-string values are typically converted to their compact JSON string representation (`null`, `true`/`false`, numbers, `"[1,2]"`, `{"k":"v"}`).
        *   **Disabled (Often Default):** Only exact matches are substituted. Strings containing placeholders but not forming an exact match are treated as literals. **This mode is required for reverse operations.**
*   **Recursive Substitution:** Recursively processes substituted values that themselves contain placeholders during the `apply` operation. The behavior within the resolved string depends on the `enableStringInterpolation` setting.
*   **Cycle Detection:** Automatically detects and prevents infinite recursion loops during the `apply` operation caused by cyclical references (e.g., `{"a": "${b}", "b": "${a}"}` or involving paths), typically throwing a specific cycle error. This works regardless of interpolation mode for lookups that are actually performed.
*   **Configurable Missing Key Handling:** Choose whether to `ignore` missing keys/paths (leaving placeholders unsubstituted, the typical default) or `error` out (throwing a specific missing key error) during `apply`.
    *   Applies to lookups triggered by exact matches in both interpolation modes.
    *   Applies to lookups triggered during interpolation *only when* `enableStringInterpolation` is `true`.
*   **Reverse Template Extraction & Application:** When string interpolation is *disabled* (`enableStringInterpolation = false`):
    *   Generate a `reverse_template` using a function like `create_reverse_template()`. This reverse template maps context variable paths to their locations (JSON Pointers) within the expected result JSON. Typically throws an error if interpolation is enabled in options.
    *   Apply this `reverse_template` to a `result_json` using a function like `apply_reverse()` to reconstruct the original data context based on the template's structure. Typically throws errors if the reverse template is malformed or if pointers don't resolve in the result JSON.
*   **Customizable Delimiters:** Define custom start (`variableStartMarker`) and end (`variableEndMarker`) markers for variables instead of the default `${` and `}`.
*   **(Implementation Specific)** Typically built using standard language features and a common JSON library for the target language.
*   **(Implementation Specific)** May include a Command-Line Tool for easy testing and usage.
*   **(Implementation Specific)** Often includes a comprehensive suite of unit tests covering core features, edge cases, and error handling.

### Why Use Permuto Templates Instead of Direct Data Structure Manipulation?

While Permuto uses standard data structure access (like JSON Pointer lookups) internally, its primary value comes from its **declarative templating approach**.

Consider transforming data from a central structure (the "context") into different JSON formats required by various APIs.

*   **Without Permuto:** You would typically write code for each target API in your chosen language. This code would programmatically build the required JSON structure and perform lookups (e.g., `context["path"]["to"]["data"]` or equivalent) to fetch data from your central structure and assign it to the correct place in the payload being built. The transformation logic resides entirely *within your application code*.
*   **With Permuto:** You define the target structure for each API in a separate **JSON template file**. These templates use `${context.path}` placeholders to specify where data from the central context should be inserted. Your application code simply loads the appropriate template and context, then calls the `apply()` function.

**Permuto's Advantage:**

*   **Declarative & Maintainable:** The structure of the API payload and the mapping from the context are defined *declaratively* in the template files. Need to adjust the format for one API? Modify its template JSON, often without changing or recompiling your application. Adding a new API? Create a new template file.
*   **Readability & Separation:** Templates clearly show the desired output structure, separating the structural definition from the application logic and the context data.
*   **Reduced Boilerplate:** Permuto handles the details of traversing the template, performing lookups, preserving data types on exact matches, optionally interpolating strings, handling missing keys gracefully, and detecting cycles – features you would otherwise need to implement manually.

In essence, Permuto provides a convenient abstraction layer for defining structure transformations through external template files rather than embedding that logic directly in application code.

## Implementation Details

For details on building, installing, testing, specific API usage, and code examples for a particular language implementation, please refer to the README file within that specific implementation's repository or directory (e.g., `README_CPP.md`, `README_PY.md`).

## Key Concepts & Examples (Conceptual)

These examples use JSON to illustrate the inputs and outputs. The core logic applies across different language implementations.

### 1. Data Type Handling (Exact Match Substitution)

This applies when a template string value consists *only* of a single placeholder (`"${...}"`). It works identically whether string interpolation is enabled or disabled during `apply`. Permuto substitutes the actual data type from the context.

| Template Fragment       | Data Context (`context`)                     | Output Fragment (`result`) | Notes                                     |
| :---------------------- | :------------------------------------------- | :------------------------- | :---------------------------------------- |
| `"num": "${data.val}"`  | `{"data": {"val": 123}}`                     | `"num": 123`               | Number type preserved.                    |
| `"bool": "${opts.on}"`  | `{"opts": {"on": true}}`                     | `"bool": true`             | Boolean type preserved.                   |
| `"arr": "${items.list}"` | `{"items": {"list": [1, "a"]}}`              | `"arr": [1, "a"]`          | Array type preserved.                     |
| `"obj": "${cfg.sec}"`   | `{"cfg": {"sec": {"k": "v"}}}`              | `"obj": {"k": "v"}`        | Object type preserved.                    |
| `"null_val": "${maybe}"`| `{"maybe": null}`                            | `"null_val": null`         | Null type preserved.                      |
| `"str_val": "${msg}"`   | `{"msg": "hello"}`                           | `"str_val": "hello"`       | String type preserved.                    |

### 2. String Interpolation (Optional Feature)

This behavior only occurs during `apply` when interpolation is **enabled** (e.g., `enableStringInterpolation = true`).

When a placeholder is part of a larger string *and* interpolation is enabled, the data value is converted to its string representation and inserted. Non-string types are typically stringified using their compact JSON representation.

**Template:**
```json
{
  "message": "User ${user.name} (ID: ${user.id}) is ${status.active}. Count: ${data.count}. Settings: ${settings}"
}
```
**Data Context:**
```json
{
  "user": {"name": "Alice", "id": 123},
  "status": {"active": true},
  "data": {"count": null},
  "settings": {"theme": "dark", "notify": false}
}
```
**Output (Interpolation Enabled):**
```json
{
  "message": "User Alice (ID: 123) is true. Count: null. Settings: {\"notify\":false,\"theme\":\"dark\"}"
}
```
**Output (Interpolation Disabled):**
```json
{
  "message": "User ${user.name} (ID: ${user.id}) is ${status.active}. Count: ${data.count}. Settings: ${settings}"
}
```
*(The entire string is treated as a literal because it's not an exact match and interpolation is disabled).*

### 3. Nested Data Access (Dot Notation)

Use dots (`.`) within placeholders to access nested properties.

**Template:** `{"city": "${address.city}", "zip": "${address.postal_code}"}`
**Data Context:** `{"address": {"city": "Anytown", "postal_code": "12345"}}`
**Output (Either Interpolation Mode):** `{"city": "Anytown", "zip": "12345"}`

*   **Path Resolution:** Implementations typically convert dot-paths (`a.b`) to the language's equivalent nested access or internal JSON Pointers (`/a/b`), handling escaping for special characters in keys (`/`, `~`, `.`) automatically based on JSON Pointer rules.
*   **Keys with Dots/Slashes/Tildes:** Accessing keys like `"user/role"` or `"db~1main"` requires the implementation to correctly parse the dot notation and apply JSON Pointer escaping rules during lookup. Using `${user/role}` should resolve to the pointer `/user~1role`.
*   **Array Indexing:** Dot notation generally does *not* support array indexing (e.g., `${arr.0}` is usually not interpreted as accessing the first element).

### 4. Recursive Substitution

If an exact match during `apply` resolves to a string containing placeholders, that string is recursively processed using the *same* `enableStringInterpolation` setting as the original call.

**Template:** ` { "msg": "${greeting.template}", "info": "${user.details}" } `
**Data Context:** ` { "greeting": { "template": "Hello, ${user.name}!" }, "user": { "name": "Bob", "details": "${user.name}" } } `

*   **Output (Interpolation Enabled):**
    *   `"${greeting.template}"` -> `"Hello, ${user.name}!"` -> (Interpolation applied) -> `"Hello, Bob!"`
    *   `"${user.details}"` -> `"${user.name}"` -> (Exact Match) -> `"Bob"`
    *   Result: `{ "msg": "Hello, Bob!", "info": "Bob" }`
*   **Output (Interpolation Disabled):**
    *   `"${greeting.template}"` -> `"Hello, ${user.name}!"` -> (No interpolation) -> Literal `"Hello, ${user.name}!"`
    *   `"${user.details}"` -> `"${user.name}"` -> (Exact Match) -> `"Bob"`
    *   Result: `{ "msg": "Hello, ${user.name}!", "info": "Bob" }`

### 5. Missing Key / Path Handling

Configure behavior using an option like `onMissingKey` (`Ignore` or `Error`).

*   **`Ignore` (Typical Default):** Unresolved placeholders remain literally (e.g., `"${missing}"` or `"Text ${missing} text"`).
*   **`Error`:** Throws a specific "missing key" exception.
    *   Triggered by failed lookups for **exact matches** (in both interpolation modes).
    *   Triggered by failed lookups during **interpolation** *only when* `enableStringInterpolation` is `true`. (Lookups aren't attempted for non-exact-match strings when interpolation is disabled).

**Template:** `{"value": "${a.b.c}", "interpolated": "Maybe: ${x.y.z}!"}`
**Data Context:** `{"a": {"b": {}}}` (`c` missing; `x` missing)

*   **`Ignore` Mode Output (Either Interpolation Mode):** `{"value": "${a.b.c}", "interpolated": "Maybe: ${x.y.z}!"}`
*   **`Error` Mode, Interpolation ON:** Throws exception when `apply` processes `"Maybe: ${x.y.z}!"`.
*   **`Error` Mode, Interpolation OFF:** Throws exception when `apply` processes `"${a.b.c}"` (exact match lookup fails). Does *not* throw for `"Maybe: ${x.y.z}!"` because interpolation is off, so the lookup isn't attempted.

### 6. Custom Delimiters

Use options like `variableStartMarker` and `variableEndMarker`. Applies in both `apply` and reverse operations.

### 7. Cycle Detection

Detects cycles like `${a}` -> `${b}` -> `${a}` during the resolution phase of `apply`. Throws a specific "cycle" exception. Applies in both interpolation modes for lookups that are actually performed.

### 8. Reverse Template Extraction & Application (Interpolation Disabled Only)

This allows recovery of the original context data if you have the `result_json` and the `original_template`, but **only** if the result was generated using `apply()` with `enableStringInterpolation = false`.

**Concept:** The `original_template` defines where context variables *should* appear in the `result_json`. A `reverse_template` is generated that maps the context variable paths to JSON Pointers indicating their location in the `result_json`.

**Process:**

1.  **Generate Reverse Template:**
    *   Call `create_reverse_template(original_template, options)` where `options.enableStringInterpolation` is `false`.
    *   This function scans `original_template` for exact-match placeholders (`"${context.path}"`).
    *   It builds a JSON structure mirroring the expected *context*, where leaf values are JSON Pointer strings pointing to the corresponding location in the *result*.
    *   Throws an error if `options.enableStringInterpolation` is `true`.

2.  **Apply Reverse Template:**
    *   Call `apply_reverse(reverse_template, result_json)`.
    *   This function traverses the `reverse_template`.
    *   For each leaf node (JSON Pointer string), it looks up the value at that pointer location within `result_json`.
    *   It builds the `reconstructed_context` data structure, placing the extracted values at the correct paths defined by the `reverse_template` structure.
    *   Throws errors if the `reverse_template` is malformed or pointers don't resolve in `result_json`.

**Example (Conceptual):**

*   **Original Template `T`:** `{"output_name": "${user.name}", "ids": ["${sys.id}"]}`
*   **Context `C`:** `{"user": {"name": "Alice"}, "sys": {"id": 1}}`
*   **Options `O`:** `enableStringInterpolation = false`
*   **Result `R = apply(T, C, O)`:** `{"output_name": "Alice", "ids": [1]}`
*   **Reverse Template `RT = create_reverse_template(T, O)`:** `{"user": {"name": "/output_name"}, "sys": {"id": "/ids/0"}}`
*   **Reconstructed Context `C_new = apply_reverse(RT, R)`:** `{"user": {"name": "Alice"}, "sys": {"id": 1}}` (Matches `C`)

**Limitations:** This extraction only works if interpolation was disabled during the original `apply()` call and only recovers context variables referenced by exact-match placeholders in the template. Any context data not used in an exact-match placeholder cannot be recovered.

## Configuration Options (Conceptual)

These options are typically passed via an `Options` object/struct or similar configuration mechanism.

| Option Name (Conceptual)  | Description                                                                   | Default Value | Typical Values               |
| :------------------------ | :---------------------------------------------------------------------------- | :------------ | :--------------------------- |
| `variableStartMarker`     | The starting delimiter for placeholders.                                      | `${`          | String                       |
| `variableEndMarker`       | The ending delimiter for placeholders.                                        | `}`           | String                       |
| `onMissingKey`            | Behavior for unresolved placeholders during `apply`.                          | `Ignore`      | `Ignore`, `Error`            |
| `enableStringInterpolation` | If true, interpolates vars in non-exact strings. If false, non-exact strings are literal. Reverse ops require false. | `false`       | Boolean (`true`/`false`)     |

*Note: Default values may vary slightly between language implementations. Check the specific implementation's documentation.*

## Command-Line Interface (CLI)

Implementations *may* provide a CLI tool, often called `permuto`. If available, it typically supports the forward `apply` operation.

**Common Synopsis:**

```bash
permuto <template_file> <context_file> [options]
```

**Common Options (Mirroring Configuration):**

*   `--on-missing-key=<value>`: `ignore` or `error`.
*   `--string-interpolation`: Flag to **enable** interpolation (often off by default).
*   `--start=<string>`: Set start delimiter.
*   `--end=<string>`: Set end delimiter.
*   `--help`: Display usage.

Refer to the specific implementation's documentation for exact CLI usage and availability.

## Contributing (General Guidelines)

Contributions to Permuto implementations are welcome! Please generally adhere to:

*   Follow standard coding practices and style guidelines for the target language.
*   Add unit tests for any new features or bug fixes, covering both interpolation modes and reverse operations where applicable.
*   Ensure all tests pass before submitting changes.
*   Consider opening an issue first to discuss significant changes.
*   Submit changes via Pull Requests to the relevant repository.

See the specific implementation's contribution guide for more details.

## License

Permuto (across its various implementations) is typically released into the **Public Domain** (Unlicense).

This means you are free to copy, modify, publish, use, compile, sell, or distribute the software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

See the `LICENSE` file in the specific implementation's repository for the full text. While attribution is not legally required, it is appreciated.
