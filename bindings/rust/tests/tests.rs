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
