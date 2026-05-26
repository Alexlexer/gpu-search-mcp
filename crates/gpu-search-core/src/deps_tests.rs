use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_root(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "gpu_search_core_deps_{name}_{}_{}",
        std::process::id(),
        unique
    ));
    fs::create_dir_all(&root).expect("temp root should be created");
    root
}

fn write(root: &Path, relative: &str, content: &str) -> DiscoveredFile {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent should be created");
    }
    fs::write(&path, content).expect("file should be written");
    DiscoveredFile {
        path,
        size: content.len() as u64,
        modified_ns: 1,
    }
}

#[test]
fn parses_python_import_and_from_import_lines() {
    let imports = parse_python_imports(
        "import os, services.user_service as users\nfrom controllers import home\nfrom .local import thing\n",
    );

    assert_eq!(
        imports,
        vec![
            PythonImport {
                module: "os".to_string()
            },
            PythonImport {
                module: "services.user_service".to_string()
            },
            PythonImport {
                module: "controllers".to_string()
            },
            PythonImport {
                module: "local".to_string()
            },
        ]
    );
}

#[test]
fn builds_python_dependency_edges_with_reasons() {
    let root = temp_root("edges");
    let service = write(&root, "service.py", "class Service: pass\n");
    let controller = write(&root, "controller.py", "import service\n");
    let readme = write(&root, "README.md", "import service\n");

    let graph = DependencyGraph::from_files(&[service.clone(), controller.clone(), readme])
        .expect("graph should build");

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from, controller.path);
    assert_eq!(graph.edges()[0].to, service.path);
    assert_eq!(graph.edges()[0].reason, "imports module service");
    fs::remove_dir_all(root).ok();
}

#[test]
fn impact_returns_reverse_dependencies_with_hops() {
    let root = temp_root("impact");
    let model = write(&root, "model.py", "class Model: pass\n");
    let service = write(&root, "service.py", "from model import Model\n");
    let controller = write(&root, "controller.py", "import service\n");

    let graph = DependencyGraph::from_files(&[model.clone(), service.clone(), controller.clone()])
        .expect("graph should build");
    let impacted = graph.impact(&model.path);

    assert_eq!(impacted.len(), 2);
    assert_eq!(impacted[0].file, service.path);
    assert_eq!(impacted[0].hops, 1);
    assert_eq!(impacted[0].reason.as_deref(), Some("imports module model"));
    assert_eq!(impacted[1].file, controller.path);
    assert_eq!(impacted[1].hops, 2);
    assert_eq!(
        impacted[1].reason.as_deref(),
        Some("imports module service")
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn unresolved_imports_do_not_create_edges() {
    let root = temp_root("unresolved");
    let app = write(&root, "app.py", "import missing_module\n");

    let graph = DependencyGraph::from_files(&[app]).expect("graph should build");

    assert!(graph.edges().is_empty());
    fs::remove_dir_all(root).ok();
}

#[test]
fn parses_js_static_and_require_imports() {
    let imports = parse_js_imports(
        "import UserService from './services/userService';\nimport './setup';\nconst auth = require(\"../auth\");\n",
    );

    assert_eq!(
        imports,
        vec![
            JsImport {
                module: "./services/userService".to_string()
            },
            JsImport {
                module: "./setup".to_string()
            },
            JsImport {
                module: "../auth".to_string()
            },
        ]
    );
}

#[test]
fn builds_js_ts_dependency_edges_with_reasons() {
    let root = temp_root("js_edges");
    let service = write(
        &root,
        "src/services/userService.ts",
        "export class UserService {}\n",
    );
    let setup = write(&root, "src/setup.js", "export const ready = true;\n");
    let controller = write(
        &root,
        "src/controllers/userController.ts",
        "import UserService from '../services/userService';\nimport '../setup';\n",
    );

    let graph = DependencyGraph::from_files(&[service.clone(), setup.clone(), controller.clone()])
        .expect("graph should build");

    assert_eq!(graph.edges().len(), 2);
    assert!(graph.edges().iter().any(|edge| {
        edge.from == controller.path
            && edge.to == service.path
            && edge.reason == "imports module ../services/userService"
    }));
    assert!(graph.edges().iter().any(|edge| {
        edge.from == controller.path
            && edge.to == setup.path
            && edge.reason == "imports module ../setup"
    }));
    fs::remove_dir_all(root).ok();
}

#[test]
fn js_impact_returns_reverse_dependencies_with_hops() {
    let root = temp_root("js_impact");
    let model = write(&root, "src/model.ts", "export class Model {}\n");
    let service = write(
        &root,
        "src/service.ts",
        "import { Model } from './model';\n",
    );
    let controller = write(
        &root,
        "src/controller.tsx",
        "const service = require('./service');\n",
    );

    let graph = DependencyGraph::from_files(&[model.clone(), service.clone(), controller.clone()])
        .expect("graph should build");
    let impacted = graph.impact(&model.path);

    assert_eq!(impacted.len(), 2);
    assert_eq!(impacted[0].file, service.path);
    assert_eq!(impacted[0].hops, 1);
    assert_eq!(
        impacted[0].reason.as_deref(),
        Some("imports module ./model")
    );
    assert_eq!(impacted[1].file, controller.path);
    assert_eq!(impacted[1].hops, 2);
    assert_eq!(
        impacted[1].reason.as_deref(),
        Some("imports module ./service")
    );
    fs::remove_dir_all(root).ok();
}

#[test]
fn resolves_js_index_imports() {
    let root = temp_root("js_index");
    let index = write(&root, "src/lib/index.ts", "export const value = 1;\n");
    let app = write(&root, "src/app.ts", "import { value } from './lib';\n");

    let graph =
        DependencyGraph::from_files(&[index.clone(), app.clone()]).expect("graph should build");

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from, app.path);
    assert_eq!(graph.edges()[0].to, index.path);
    assert_eq!(graph.edges()[0].reason, "imports module ./lib");
    fs::remove_dir_all(root).ok();
}

#[test]
fn parses_csharp_namespace_usings_and_type_declarations() {
    let info = parse_csharp_file(
        "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic class UserController : BaseController, IUserController { }\n",
    );

    assert_eq!(info.namespace.as_deref(), Some("MyApp.Controllers"));
    assert_eq!(info.usings, vec!["MyApp.Services"]);
    assert_eq!(info.declared_types, vec!["UserController"]);
    assert_eq!(info.base_types, vec!["BaseController", "IUserController"]);
}

#[test]
fn builds_csharp_using_namespace_edge() {
    let root = temp_root("cs_using");
    let service = write(
        &root,
        "Services/UserService.cs",
        "namespace MyApp.Services;\npublic class UserService { }\n",
    );
    let controller = write(
        &root,
        "Controllers/UserController.cs",
        "using MyApp.Services;\nnamespace MyApp.Controllers;\npublic class UserController { }\n",
    );

    let graph = DependencyGraph::from_files(&[service.clone(), controller.clone()])
        .expect("graph should build");

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from, controller.path);
    assert_eq!(graph.edges()[0].to, service.path);
    assert_eq!(graph.edges()[0].reason, "imports namespace MyApp.Services");
    fs::remove_dir_all(root).ok();
}

#[test]
fn builds_csharp_type_reference_edge_with_reason() {
    let root = temp_root("cs_type_ref");
    let service = write(
        &root,
        "Services/UserService.cs",
        "namespace MyApp.Services;\npublic class UserService { }\n",
    );
    let controller = write(
        &root,
        "Controllers/UserController.cs",
        "namespace MyApp.Controllers;\npublic class UserController { private UserService _service; }\n",
    );

    let graph = DependencyGraph::from_files(&[service.clone(), controller.clone()])
        .expect("graph should build");

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from, controller.path);
    assert_eq!(graph.edges()[0].to, service.path);
    assert_eq!(graph.edges()[0].reason, "references type UserService");
    fs::remove_dir_all(root).ok();
}

#[test]
fn csharp_base_and_interface_reasons_take_priority() {
    let root = temp_root("cs_base_interface");
    let base = write(
        &root,
        "Base/BaseService.cs",
        "namespace MyApp.Base;\npublic class BaseService { }\n",
    );
    let interface = write(
        &root,
        "Contracts/IUserService.cs",
        "namespace MyApp.Contracts;\npublic interface IUserService { }\n",
    );
    let service = write(
        &root,
        "Services/UserService.cs",
        "using MyApp.Base;\nusing MyApp.Contracts;\nnamespace MyApp.Services;\npublic class UserService : BaseService, IUserService { }\n",
    );

    let graph = DependencyGraph::from_files(&[base.clone(), interface.clone(), service.clone()])
        .expect("graph should build");

    assert_eq!(graph.edges().len(), 2);
    assert!(graph.edges().iter().any(|edge| {
        edge.from == service.path
            && edge.to == base.path
            && edge.reason == "inherits from BaseService"
    }));
    assert!(graph.edges().iter().any(|edge| {
        edge.from == service.path
            && edge.to == interface.path
            && edge.reason == "implements interface IUserService"
    }));
    fs::remove_dir_all(root).ok();
}

#[test]
fn csharp_impact_returns_reason_and_hops() {
    let root = temp_root("cs_impact");
    let service = write(
        &root,
        "Services/UserService.cs",
        "namespace MyApp.Services;\npublic class UserService { }\n",
    );
    let controller = write(
        &root,
        "Controllers/UserController.cs",
        "namespace MyApp.Controllers;\npublic class UserController { private UserService _service; }\n",
    );

    let graph = DependencyGraph::from_files(&[service.clone(), controller.clone()])
        .expect("graph should build");
    let impacted = graph.impact(&service.path);

    assert_eq!(impacted.len(), 1);
    assert_eq!(impacted[0].file, controller.path);
    assert_eq!(impacted[0].hops, 1);
    assert_eq!(
        impacted[0].reason.as_deref(),
        Some("references type UserService")
    );
    fs::remove_dir_all(root).ok();
}
