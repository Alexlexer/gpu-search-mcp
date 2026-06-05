use gpu_search_core::{discover_files, PatternSearchOptions, search_files, IndexOptions};
use std::fs;

#[test]
fn integration_parity_simple_search() {
	let root = std::env::temp_dir().join(format!("gpu_search_integ_{}", std::process::id()));
	fs::create_dir_all(&root).unwrap();
	let file = root.join("sample.rs");
	fs::write(&file, "fn main() { println!(\"hello\"); }\nlet user_service = UserService::new();\n").unwrap();

	let opts = IndexOptions::default();
	let files = discover_files(&root, &opts).expect("discover files");

	let options = PatternSearchOptions { max_results: 10, ..PatternSearchOptions::default() };
	let results = search_files(&files, "UserService", &options).expect("search files");

	assert_eq!(results.len(), 1);
	assert!(results[0].snippet.contains("UserService"));

	fs::remove_dir_all(&root).ok();
}
