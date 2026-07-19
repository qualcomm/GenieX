use geniex::*;

#[test]
fn test_version() {
    let _v = version();
}

#[test]
fn test_error() {
    let err = GeniexError::CommonInvalidInput;
    assert_ne!(err.message(), "");
    let res = GeniexError::check(0);
    assert!(res.is_ok());
}

#[test]
fn test_config_defaults() {
    let model_cfg = ModelConfig::default();
    assert_eq!(model_cfg.n_ctx, 0);

    let sampler_cfg = SamplerConfig::default();
    assert_eq!(sampler_cfg.seed, -1);

    let gen_cfg = GenerationConfig::default();
    assert_eq!(gen_cfg.max_tokens, 0);
}

#[test]
fn test_chat_message() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
    };
    assert_eq!(msg.role, "user");
}

#[test]
fn test_init_and_plugins() -> Result<()> {
    init()?;
    let ver = version();
    assert!(!ver.is_empty(), "Version string should not be empty");

    let plugins = get_plugin_list()?;
    println!("Loaded plugins: {:?}", plugins);

    deinit()?;
    Ok(())
}

#[test]
fn test_resolve_device() -> Result<()> {
    let input = ResolveDeviceInput {
        plugin_id: "llama_cpp".to_string(),
        model_name: Some("test.gguf".to_string()),
        mode: Some("cpu".to_string()),
        ngl_default: -1,
    };
    let output = resolve_device(&input)?;
    assert_eq!(output.ngl, 0); // cpu forces ngl = 0
    Ok(())
}
