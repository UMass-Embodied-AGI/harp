"""Static regression tests for the frozen v1 release compatibility surface.

These tests deliberately parse source instead of importing ``envs.tasks`` or
the CLI modules.  Importing the registry imports Genesis and every task class;
none of that runtime work is needed to detect a public-name or schema drift.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "compat" / "release_contract_v1.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
_MISSING = object()


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_assignment(tree: ast.AST, name: str) -> ast.AST:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    raise AssertionError(f"assignment {name!r} was not found")


def _literal_assignment(relative_path: str, name: str):
    return ast.literal_eval(_find_assignment(_tree(relative_path), name))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} was not found")


def _find_class_method(
    tree: ast.AST, class_name: str, method_name: str
) -> ast.FunctionDef:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == method_name:
                return statement
        raise AssertionError(
            f"method {class_name}.{method_name} was not found"
        )
    raise AssertionError(f"class {class_name!r} was not found")


def _class_literal_attribute(
    tree: ast.AST, class_name: str, attribute: str
):
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        return ast.literal_eval(_find_assignment(node, attribute))
    raise AssertionError(f"class {class_name!r} was not found")


def _literal_dict_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Dict):
            continue
        for key in candidate.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _stored_subscript_keys(node: ast.AST, object_name: str) -> set[str]:
    keys: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript) or not isinstance(candidate.ctx, ast.Store):
            continue
        if not isinstance(candidate.value, ast.Name) or candidate.value.id != object_name:
            continue
        if isinstance(candidate.slice, ast.Constant) and isinstance(candidate.slice.value, str):
            keys.add(candidate.slice.value)
    return keys


class _StaticClassIndex:
    """Resolve class attributes and inheritance without importing a module."""

    def __init__(self) -> None:
        self.classes: dict[str, tuple[str, ast.ClassDef]] = {}
        self.imports: dict[str, dict[str, str]] = {}
        for path in sorted((ROOT / "envs").rglob("*.py")):
            relative = path.relative_to(ROOT)
            parts = list(relative.with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            module = ".".join(parts)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: dict[str, str] = {}
            package_parts = module.split(".") if is_package else module.split(".")[:-1]
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        keep = len(package_parts) - (node.level - 1)
                        base_parts = package_parts[:keep]
                        if node.module:
                            base_parts.extend(node.module.split("."))
                        base = ".".join(base_parts)
                    else:
                        base = node.module or ""
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        local = alias.asname or alias.name
                        imports[local] = f"{base}.{alias.name}" if base else alias.name
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = alias.name
                elif isinstance(node, ast.ClassDef):
                    self.classes[f"{module}.{node.name}"] = (module, node)
            self.imports[module] = imports

    def reference(self, module: str, expression: ast.AST) -> str | None:
        if isinstance(expression, ast.Name):
            return self.imports[module].get(expression.id, f"{module}.{expression.id}")
        if isinstance(expression, ast.Attribute):
            pieces = []
            cursor = expression
            while isinstance(cursor, ast.Attribute):
                pieces.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                root = self.imports[module].get(cursor.id, f"{module}.{cursor.id}")
                return ".".join([root, *reversed(pieces)])
        return None

    @staticmethod
    def _direct_attribute(node: ast.ClassDef, attribute: str):
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                if any(
                    isinstance(target, ast.Name) and target.id == attribute
                    for target in statement.targets
                ):
                    return ast.literal_eval(statement.value)
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == attribute
            ):
                return ast.literal_eval(statement.value)
        return _MISSING

    def attribute(self, qualified_class: str, attribute: str, seen=None):
        seen = set() if seen is None else set(seen)
        if qualified_class in seen or qualified_class not in self.classes:
            raise KeyError(f"cannot statically resolve {qualified_class}.{attribute}")
        seen.add(qualified_class)
        module, node = self.classes[qualified_class]
        value = self._direct_attribute(node, attribute)
        if value is not _MISSING:
            return value
        for base_expression in node.bases:
            base = self.reference(module, base_expression)
            if base not in self.classes:
                continue
            try:
                return self.attribute(base, attribute, seen)
            except KeyError:
                continue
        raise KeyError(f"cannot statically resolve {qualified_class}.{attribute}")


CLASS_INDEX = _StaticClassIndex()


def _descriptor_records() -> list[dict[str, object]]:
    """Read the authoritative descriptor data without exercising its builder."""

    module = "envs.tasks.registry"
    node = _find_assignment(_tree("envs/tasks/registry.py"), "TASK_DESCRIPTORS")
    assert isinstance(node, (ast.Tuple, ast.List))
    records = []
    for descriptor in node.elts:
        assert isinstance(descriptor, ast.Call)
        assert isinstance(descriptor.func, ast.Name) and descriptor.func.id == "_descriptor"
        assert len(descriptor.args) >= 4
        name = ast.literal_eval(descriptor.args[0])
        class_expression = descriptor.args[1]
        qualified = CLASS_INDEX.reference(module, class_expression)
        assert qualified is not None
        class_module, class_name = qualified.rsplit(".", 1)
        family = ast.literal_eval(descriptor.args[2])
        variant_node = descriptor.args[3]
        assert isinstance(variant_node, ast.Attribute)
        variant = variant_node.attr.lower()
        keywords = {keyword.arg: keyword.value for keyword in descriptor.keywords}
        registration_node = keywords.get("registration")
        if registration_node is None:
            registration = "canonical"
        else:
            assert isinstance(registration_node, ast.Attribute)
            registration = registration_node.attr.lower()
        compatibility = registration == "compatibility"
        variant_label_node = keywords.get("variant_label")
        variant_label = (
            ast.literal_eval(variant_label_node)
            if variant_label_node is not None
            else None
        )
        no_human_node = keywords.get("no_human_name")
        no_human_name = ast.literal_eval(no_human_node) if no_human_node is not None else None
        canonical_node = keywords.get("no_human_canonical")
        no_human_canonical = (
            bool(ast.literal_eval(canonical_node)) if canonical_node is not None else False
        )
        records.append(
            {
                "name": name,
                "module": class_module,
                "class": class_name,
                "family": family,
                "variant": variant,
                "registration": registration,
                "variant_label": variant_label,
                "compatibility": compatibility,
                "no_human_name": no_human_name,
                "no_human_canonical": no_human_canonical,
            }
        )
    return records


def _registry_class_map(assignment: str) -> dict[str, dict[str, str]]:
    records = _descriptor_records()
    if assignment == "TASK_MAP":
        selected = ((record["name"], record) for record in records if not record["compatibility"])
    elif assignment == "TASK_ALIASES":
        selected = ((record["name"], record) for record in records if record["compatibility"])
    elif assignment == "NO_HUMAN_CANONICAL_TASK_MAP":
        selected = (
            (record["no_human_name"], record)
            for record in records
            if record["no_human_canonical"]
        )
    else:
        raise AssertionError(f"unsupported registry projection {assignment}")
    return {
        str(name): {"module": str(record["module"]), "class": str(record["class"])}
        for name, record in selected
    }


def _descriptor_no_human_aliases() -> dict[str, str]:
    return {
        str(record["name"]): str(record["no_human_name"])
        for record in _descriptor_records()
        if record["no_human_name"] is not None
        and record["name"] != record["no_human_name"]
    }


def _descriptor_metadata() -> dict[str, dict[str, object]]:
    return {
        str(record["name"]): {
            "family": str(record["family"]),
            "variant": str(record["variant"]),
            "registration": str(record["registration"]),
            "variant_label": record["variant_label"],
        }
        for record in _descriptor_records()
    }


def _identity_only(entries: dict) -> dict:
    return {
        name: {"module": entry["module"], "class": entry["class"]}
        for name, entry in entries.items()
    }


def _default_instruction(qualified_class: str) -> tuple[str, str]:
    template = CLASS_INDEX.attribute(qualified_class, "INSTRUCTION")
    assert isinstance(template, str) and template
    if "{object}" not in template:
        return template, template

    object_set = CLASS_INDEX.attribute(qualified_class, "OBJECT_SET")
    catalog_tree = _tree("envs/object_catalog.py")
    categories = ast.literal_eval(_find_assignment(catalog_tree, "CATEGORIES"))
    tokens = categories.get(object_set, [object_set]) if isinstance(object_set, str) else object_set
    assert tokens, f"empty OBJECT_SET for {qualified_class}"
    object_key = str(tokens[0]).partition("@")[0]

    entries_node = _find_assignment(catalog_tree, "_ENTRIES")
    assert isinstance(entries_node, (ast.Tuple, ast.List))
    display_names = {}
    for call in entries_node.elts:
        if not isinstance(call, ast.Call) or not call.args:
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "ObjectEntry":
            continue
        key = ast.literal_eval(call.args[0])
        display = next(
            (ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg == "display_name"),
            "",
        )
        display_names[key] = display or key
    return template, template.format(object=display_names[object_key])


def test_contract_file_is_versioned_readable_and_deterministically_ordered():
    assert CONTRACT["contract_schema"] == "genesis_hr_bench.release_contract.v1"
    assert CONTRACT_PATH.read_text(encoding="utf-8").endswith("\n")
    ordered_maps = (
        CONTRACT["task_registry"]["canonical"],
        CONTRACT["task_registry"]["accepted_aliases"],
        CONTRACT["task_registry"]["descriptor_metadata"],
        CONTRACT["no_human"]["canonical"],
        CONTRACT["no_human"]["variant_aliases"],
        CONTRACT["model_registry"]["policies"],
        CONTRACT["model_registry"]["deployable"],
        CONTRACT["script_cli"],
    )
    for mapping in ordered_maps:
        assert list(mapping) == sorted(mapping)


def test_frozen_task_names_and_exact_class_identities_match_source():
    frozen = CONTRACT["task_registry"]
    records = _descriptor_records()
    canonical_order = [
        str(record["name"])
        for record in records
        if not record["compatibility"]
    ]
    # HARP-Bench names: 49 canonical; the 32 pre-HARP names are aliases.
    assert frozen["canonical_count"] == 49
    assert frozen["alias_count"] == 32
    assert len(frozen["canonical"]) == 49
    assert len(frozen["accepted_aliases"]) == 32
    assert canonical_order == frozen["canonical_order"]
    assert set(frozen["canonical_order"]) == set(frozen["canonical"])
    assert _registry_class_map("TASK_MAP") == _identity_only(frozen["canonical"])
    assert _registry_class_map("TASK_ALIASES") == _identity_only(frozen["accepted_aliases"])
    assert set(frozen["canonical"]).isdisjoint(frozen["accepted_aliases"])


def test_frozen_descriptor_semantics_match_authoritative_manifest():
    frozen = CONTRACT["task_registry"]
    metadata = frozen["descriptor_metadata"]

    assert len(metadata) == frozen["canonical_count"] + frozen["alias_count"]
    assert set(metadata) == {*frozen["canonical"], *frozen["accepted_aliases"]}
    assert _descriptor_metadata() == metadata


def test_every_frozen_task_instruction_matches_static_class_contract():
    frozen = {
        **CONTRACT["task_registry"]["canonical"],
        **CONTRACT["task_registry"]["accepted_aliases"],
    }
    for task_name, entry in frozen.items():
        qualified = f"{entry['module']}.{entry['class']}"
        actual_template, actual_instruction = _default_instruction(qualified)
        assert actual_instruction == entry["instruction"], task_name
        assert actual_template == entry.get("instruction_template", entry["instruction"]), task_name
        assert "{object}" not in entry["instruction"], task_name


def test_frozen_no_human_canonical_and_variant_mappings_match_source():
    frozen = CONTRACT["no_human"]
    assert _registry_class_map("NO_HUMAN_CANONICAL_TASK_MAP") == frozen["canonical"]
    assert _descriptor_no_human_aliases() == frozen["variant_aliases"]
    assert all(
        canonical in frozen["canonical"]
        for canonical in frozen["variant_aliases"].values()
    )
    assert "categorize" in frozen["canonical"]
    assert "categorize" not in CONTRACT["task_registry"]["canonical"]
    assert "categorize_objects_interruption" in CONTRACT["task_registry"]["canonical"]
    assert "categorize_interrupt" in CONTRACT["task_registry"]["accepted_aliases"]


def test_core_task_api_method_names_remain_available():
    # BaseTask's implementation is split across envs/base_task.py and the
    # mixin modules in envs/base_task_parts/; the contract holds over the
    # union of their class bodies (still a source-level check, no import).
    methods = set()
    sources = ["envs/base_task.py"] + sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "envs/base_task_parts").glob("*.py")
    )
    for source in sources:
        for node in _tree(source).body:
            if isinstance(node, ast.ClassDef) and (
                node.name == "BaseTask" or node.name.endswith("Mixin")
            ):
                methods |= {
                    sub.name
                    for sub in node.body
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
    assert set(CONTRACT["task_api"]["required_methods"]) <= methods
    expert_source = (ROOT / "scripts/eval_expert_set.py").read_text(encoding="utf-8")
    assert CONTRACT["task_api"]["optional_expert_hook"] in expert_source


def _model_registry_from_ast() -> dict[str, str]:
    node = _find_assignment(_tree("scripts/vla_client.py"), "VLA_REGISTRY")
    assert isinstance(node, ast.Dict)
    result = {}
    for key, value in zip(node.keys, node.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert isinstance(value, ast.Name)
        result[key.value] = value.id
    return result


def _wrapper_model_contract() -> dict[str, dict[str, object]]:
    source = (ROOT / "scripts/run_vla_eval.sh").read_text(encoding="utf-8")
    start = source.index('case "$MODEL" in')
    end = source.index('\nesac\n\nPORT=', start)
    body = source[start + len('case "$MODEL" in') : end]
    arms = re.findall(
        r"(?ms)^  ([a-z0-9_*|]+)\)\n(.*?)(?=^  [a-z0-9_*|]+\)\n|\Z)",
        body,
    )
    result: dict[str, dict[str, object]] = {}
    for label, arm in arms:
        if label == "*":
            continue
        script_match = re.search(r'SERVER_SCRIPT="([^"]+)"', arm)
        action_match = re.search(r'DEFAULT_ACTION_TYPE="([^"]+)"', arm)
        assert script_match and action_match, label
        models = label.split("|")
        nested_ports = {
            name: int(port)
            for name, port in re.findall(
                r"(?m)^      ([a-z0-9_]+)\)\s+DEFAULT_PORT=(\d+)\s*;;", arm
            )
        }
        direct_port = re.search(r"(?m)^    DEFAULT_PORT=(\d+)\s*$", arm)
        for model in models:
            port = nested_ports.get(model)
            if port is None:
                assert direct_port, model
                port = int(direct_port.group(1))
            result[model] = {
                "action_type": action_match.group(1),
                "port": port,
                "server_script": script_match.group(1),
            }
    return result


def test_frozen_model_names_classes_and_wrapper_dispatch_match_source():
    frozen = CONTRACT["model_registry"]
    assert _model_registry_from_ast() == frozen["policies"]
    assert _wrapper_model_contract() == frozen["deployable"]
    assert set(frozen["deployable"]) < set(frozen["policies"])


def test_frozen_eval_set_and_scene_init_versions_and_keys_match_source():
    eval_contract = CONTRACT["schemas"]["eval_set"]
    assert _literal_assignment(eval_contract["source"], "SCHEMA_VERSION") == eval_contract["version"]
    defaults = _literal_assignment(eval_contract["source"], "DEFAULTS")
    base_episodes = _literal_assignment(eval_contract["source"], "BASE_EPISODES")
    episode_keys = set().union(*(episode.keys() for episode in base_episodes))
    eval_tree = _tree(eval_contract["source"])
    actual_jsonl_keys = (
        set(defaults)
        | episode_keys
        | _literal_dict_keys(_find_function(eval_tree, "with_ids"))
        | _literal_dict_keys(_find_function(eval_tree, "build_jsonl"))
    )
    assert actual_jsonl_keys == set(eval_contract["jsonl_required_keys"])

    scene_contract = CONTRACT["schemas"]["scene_init"]
    assert _literal_assignment(scene_contract["source"], "SCHEMA_VERSION") == scene_contract["version"]
    capture = _find_function(_tree(scene_contract["source"]), "capture_scene_init")
    capture_keys = _literal_dict_keys(capture) | _stored_subscript_keys(capture, "init")
    assert set(scene_contract["required_keys"]) <= capture_keys


def test_manifest_loaders_enforce_the_frozen_required_key_surface():
    manifest_source = "envs/manifest_io.py"
    eval_contract = CONTRACT["schemas"]["eval_set"]
    scene_contract = CONTRACT["schemas"]["scene_init"]

    assert list(_literal_assignment(manifest_source, "EVAL_SET_REQUIRED_KEYS")) == eval_contract[
        "jsonl_required_keys"
    ]
    assert list(_literal_assignment(manifest_source, "SCENE_INIT_REQUIRED_KEYS")) == scene_contract[
        "required_keys"
    ]
    assert list(
        _literal_assignment(manifest_source, "SCENE_INIT_SET_REQUIRED_KEYS")
    ) == scene_contract["set_required_keys"]


def test_frozen_interaction_serialization_schemas_match_source():
    for contract_name, method_name in (
        ("policy_directive", "to_dict"),
        ("interaction_state", "state_dict"),
        ("block_ranking_assist_state", "state_dict"),
        ("block_ranking_interrupt_state", "state_dict"),
    ):
        contract = CONTRACT["schemas"][contract_name]
        tree = _tree(contract["source"])
        method = _find_class_method(
            tree, contract["source_class"], method_name
        )
        assert _literal_dict_keys(method) == set(contract["required_keys"])
        if contract_name.startswith("block_ranking_"):
            observed_version = _class_literal_attribute(
                tree, contract["source_class"], "STATE_SCHEMA"
            )
            assert observed_version == contract["version"]
        else:
            string_literals = {
                node.value
                for node in ast.walk(method)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            assert contract["version"] in string_literals


def test_frozen_vla_h5_and_legacy_pickle_schemas_match_source():
    vla_contract = CONTRACT["schemas"]["vla_steps"]
    assert _literal_assignment(vla_contract["source"], "SCHEMA_VERSION") == vla_contract["version"]
    assert list(_literal_assignment(vla_contract["source"], "ACTION_LABEL_KEYS")) == vla_contract[
        "action_label_keys"
    ]
    write_h5 = _find_function(_tree(vla_contract["source"]), "_write_h5")
    datasets = set()
    groups = set()
    attributes = set()
    for node in ast.walk(write_h5):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                if node.func.attr == "create_dataset":
                    datasets.add(node.args[0].value)
                elif node.func.attr == "create_group":
                    groups.add(node.args[0].value)
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
            continue
        if not isinstance(node.value, ast.Attribute) or node.value.attr != "attrs":
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            attributes.add(node.slice.value)
    assert datasets == set(vla_contract["required_h5_datasets"])
    assert groups == set(vla_contract["h5_groups"])
    assert attributes == set(vla_contract["h5_attributes"])

    pickle_contract = CONTRACT["schemas"]["legacy_trajectory_pickle"]
    save = _find_function(_tree(pickle_contract["source"]), "save_trajectory")
    payload = next(
        node.value
        for node in save.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
    )
    assert isinstance(payload, ast.Dict)
    assert _literal_dict_keys(payload) == set(pickle_contract["required_keys"])
    assert _stored_subscript_keys(save, "payload") == set(pickle_contract["optional_keys"])


def test_frozen_rlds_version_name_and_feature_keys_match_source():
    rlds_contract = CONTRACT["schemas"]["rlds"]
    entrypoint = _tree(rlds_contract["source"])
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "scripts.data_conversion"
        and any(alias.name == "rlds" and alias.asname == "_implementation"
                for alias in node.names)
        for node in entrypoint.body
    ), "the frozen public entry point must forward to the RLDS implementation"
    tree = _tree("scripts/data_conversion/rlds.py")
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == rlds_contract["builder_class"]
    )
    version_value = _find_assignment(builder, "VERSION")
    assert isinstance(version_value, ast.Call) and version_value.args
    assert ast.literal_eval(version_value.args[0]) == rlds_contract["version"]
    snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", builder.name).lower()
    assert snake_name == rlds_contract["dataset_name"]
    feature_keys = _literal_dict_keys(_find_function(builder, "_info"))
    assert set(rlds_contract["observation_keys"]) <= feature_keys
    assert set(rlds_contract["step_keys"]) <= feature_keys
    assert set(rlds_contract["episode_metadata_keys"]) <= feature_keys


def _static_cli_arguments(relative_path: str) -> tuple[dict[str, dict], set[str]]:
    tree = _tree(relative_path)
    arguments: dict[str, dict] = {}
    parse_modes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"parse_args", "parse_known_args"}:
            parse_modes.add(node.func.attr)
        if node.func.attr != "add_argument":
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        if not flags:
            continue
        spec = {"aliases": flags[1:]}
        for keyword in node.keywords:
            if keyword.arg not in {"required", "default", "choices", "action"}:
                continue
            try:
                spec[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
        arguments[flags[0]] = spec
    return arguments, parse_modes


def test_representative_frozen_cli_arguments_match_static_parsers():
    for script, script_contract in CONTRACT["script_cli"].items():
        actual, parse_modes = _static_cli_arguments(script)
        assert script_contract["parse_mode"] in parse_modes, script
        for flag, expected in script_contract["arguments"].items():
            assert flag in actual, f"{script}: missing {flag}"
            for key, value in expected.items():
                assert actual[flag].get(key) == value, f"{script}: {flag} {key}"


def test_all_direct_release_requirements_are_exactly_pinned():
    unpinned = []
    for relative_path in (
        "requirements-genesis-latest.txt",
        "requirements.txt",
        "requirements-dev.txt",
    ):
        for line_number, raw_line in enumerate(
            (ROOT / relative_path).read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            requirement = raw_line.split("#", 1)[0].strip()
            if not requirement or requirement.startswith(("-r ", "--")):
                continue
            if "==" not in requirement:
                unpinned.append(f"{relative_path}:{line_number}: {requirement}")
    assert unpinned == []
