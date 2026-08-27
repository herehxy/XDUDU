//! 根据已解析配置创建具体 Provider，避免 CLI 依赖厂商实现细节。

use std::time::Duration;

use super::{
    AnthropicProvider, DeepSeekProvider, OpenAiCompatibleProvider, Provider, RetryingProvider,
};
use crate::{
    config::ProviderConfig,
    credentials::SecretString,
    error::{ErrorKind, XduduError, XduduResult},
};

pub trait ProviderFactory {
    fn create(
        &self,
        config: &ProviderConfig,
        secret: SecretString,
    ) -> XduduResult<Box<dyn Provider>>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultProviderFactory;

impl ProviderFactory for DefaultProviderFactory {
    fn create(
        &self,
        config: &ProviderConfig,
        secret: SecretString,
    ) -> XduduResult<Box<dyn Provider>> {
        let timeout = Duration::from_secs(config.timeout_seconds);
        let api_key = secret.into_exposed();
        let provider: Box<dyn Provider> = match config.name.as_str() {
            "anthropic" => Box::new(AnthropicProvider::with_timeout(
                api_key,
                config
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.anthropic.com"),
                timeout,
            )?),
            "deepseek" => Box::new(DeepSeekProvider::with_timeout(
                api_key,
                config
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.deepseek.com"),
                timeout,
            )?),
            "openai-compatible" => Box::new(OpenAiCompatibleProvider::with_timeout(
                api_key,
                config
                    .base_url
                    .as_deref()
                    .unwrap_or("https://api.openai.com/v1"),
                timeout,
            )?),
            other => {
                return Err(XduduError::new(
                    ErrorKind::ConfigError,
                    format!("不支持的 Provider：{other}"),
                ));
            }
        };
        Ok(Box::new(RetryingProvider::with_interval(
            provider,
            config.max_attempts,
            Duration::from_millis(config.retry_base_ms),
            Duration::from_millis(config.min_request_interval_ms),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 工厂创建配置指定的_provider() {
        let config = ProviderConfig {
            name: "deepseek".into(),
            model: "deepseek-chat".into(),
            base_url: Some("http://127.0.0.1:1234".into()),
            timeout_seconds: 30,
            max_attempts: 3,
            retry_base_ms: 10,
            min_request_interval_ms: 0,
            temperature: 0.2,
            max_output_tokens: 4096,
            reasoning: false,
            max_context_tokens: 0,
        };
        let provider = DefaultProviderFactory
            .create(&config, SecretString::new("test-key").unwrap())
            .unwrap();
        assert_eq!(provider.name(), "deepseek");
    }

    #[test]
    fn 工厂创建_openai_compatible() {
        let config = ProviderConfig {
            name: "openai-compatible".into(),
            model: "gpt-5".into(),
            base_url: Some("http://127.0.0.1:1235".into()),
            timeout_seconds: 30,
            max_attempts: 3,
            retry_base_ms: 10,
            min_request_interval_ms: 0,
            temperature: 0.2,
            max_output_tokens: 4096,
            reasoning: false,
            max_context_tokens: 0,
        };
        let provider = DefaultProviderFactory
            .create(&config, SecretString::new("test-key").unwrap())
            .unwrap();
        assert_eq!(provider.name(), "openai-compatible");
    }
}
