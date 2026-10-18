#!/usr/bin/env python3
import glob
import os
import re


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all *_model.py files except the output file we’re going to create
    model_files = sorted(
        f
        for f in glob.glob(os.path.join(base_dir, "*_model.py"))
        if os.path.basename(f) != "model.py"
    )

    # Make sure meta_model.py goes last (it depends on the others)
    model_files = sorted(
        model_files,
        key=lambda p: (os.path.basename(p) == "meta_model.py", os.path.basename(p)),
    )

    module_names = {
        os.path.splitext(os.path.basename(path))[0] for path in model_files
    }

    import_lines: list[str] = []
    code_chunks: list[str] = []

    for path in model_files:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        local_code: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("import ") or stripped.startswith("from "):
                # Detect whether this import is internal (another *_model module)
                is_internal = False

                # from foo_model import Bar
                m_from = re.match(r"from\s+([a-zA-Z0-9_.]+)\s+import", stripped)
                # import foo_model, bar_model
                m_import = re.match(r"import\s+(.+)", stripped)

                if m_from:
                    mod = m_from.group(1).lstrip(".")  # handle optional leading dot
                    if mod in module_names:
                        is_internal = True
                elif m_import:
                    mods_raw = m_import.group(1)
                    mods = [m.strip().lstrip(".") for m in mods_raw.split(",")]
                    # treat as internal if *all* imported modules are our *_model modules
                    if all(m in module_names for m in mods):
                        is_internal = True

                if not is_internal and line not in import_lines:
                    import_lines.append(line)
            else:
                local_code.append(line)

        chunk_header = f"# ===== BEGIN {os.path.basename(path)} =====\n"
        chunk_footer = f"# ===== END {os.path.basename(path)} =====\n\n"
        code_chunks.append(chunk_header + "".join(local_code) + chunk_footer)

    out_path = os.path.join(base_dir, "model.py")
    with open(out_path, "w", encoding="utf-8") as out:
        # Write all unique imports
        for line in import_lines:
            out.write(line)
        out.write("\n\n")
        # Then all code blocks
        for chunk in code_chunks:
            out.write(chunk)

    print(f"Created {out_path} from:")
    for p in model_files:
        print("  -", os.path.basename(p))


if __name__ == "__main__":
    main()
