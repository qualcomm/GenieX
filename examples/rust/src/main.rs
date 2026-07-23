use geniex::*;
use std::env;

fn main() -> Result<()> {
    println!("=== GenieX Rust Binding Functional Example ===");

    init()?;
    println!("[+] SDK Initialized successfully.");
    println!("[+] GenieX Version: {}", version());

    let plugins = get_plugin_list()?;
    println!("[+] Installed plugins: {:?}", plugins);

    if plugins.contains(&"llama_cpp".to_string()) {
        let resolve_input = ResolveDeviceInput {
            plugin_id: "llama_cpp".to_string(),
            model_name: Some("example.gguf".to_string()),
            mode: Some("cpu".to_string()),
            ngl_default: -1,
        };
        let dev_output = resolve_device(&resolve_input)?;
        println!(
            "[+] Device resolution: device_id={:?}, ngl={}",
            dev_output.device_id, dev_output.ngl
        );
        if let Some(warn) = dev_output.warning {
            println!("[!] Device warning: {}", warn);
        }
    }

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let model_path = &args[1];
        println!("\n[+] Loading LLM model from: {}", model_path);

        let config = ModelConfig::default();
        let mut llm = Llm::create(
            model_path,
            "llama_cpp",
            &config,
            None,
            None,
            None,
        )?;

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello! Describe what GenieX is in one sentence.".to_string(),
        }];

        println!("[+] Applying chat template...");
        let prompt = llm.apply_chat_template(&messages, None, false, true)?;

        println!("[+] Generating response...");
        let (response_text, _profile) =
            llm.generate::<fn(&str) -> bool>(Some(&prompt), None, None, None)?;

        println!("\n--- Generated Response ---");
        println!("{}", response_text);
        println!("--------------------------");
    } else {
        println!("\n[i] Note: Pass a GGUF model file path as an argument to run LLM inference:");
        println!("    cargo run -- <path_to_model.gguf>");
    }

    deinit()?;
    println!("\n[+] SDK De-initialized successfully.");

    Ok(())
}
