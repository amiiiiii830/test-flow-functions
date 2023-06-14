use chrono::{Duration, Utc};
use dotenv::dotenv;
use http_req::{request, request::Method, request::Request, uri::Uri};
use openai_flows::{
    chat::{ChatModel, ChatOptions},
    OpenAIFlows,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use slack_flows::{listen_to_channel, send_message_to_channel, SlackMessage};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};
use tiktoken_rs::cl100k_base;
use tokio;
use web_scraper_flows::get_page_text;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() {
    dotenv().ok();

    let slack_workspace = env::var("slack_workspace").unwrap_or("secondstate".to_string());
    let slack_channel = env::var("slack_channel").unwrap_or("github-status".to_string());

    listen_to_channel(&slack_workspace, &slack_channel, |sm| {
        handler(&slack_workspace, &slack_channel, sm.text)
    })
    .await;
}

async fn handler(workspace: &str, channel: &str, text: String) {
    let flow_test_trigger = "hacker";
    let private_test_trigger = "private";

    if let Ok(_) = Uri::try_from(text.as_ref()) {
        if let Some(clean_text) = test_scraper_integration(&text).await {
            test_openai_integration_summary(clean_text).await;
            return;
        }
    }

    if text.starts_with(private_test_trigger) {
        let mut openai = OpenAIFlows::new();
        openai.set_retry_times(3);

        let chat_id = format!("converstation N");
        let system = &format!("You're a chatbot.");

        let co = ChatOptions {
            model: ChatModel::GPT35Turbo,
            restart: true,
            system_prompt: Some(system),
        };

        let msg_text = text
            .split_whitespace()
            .skip(1)
            .collect::<Vec<&str>>()
            .join(" ");
        let question = format!("given user input: {msg_text}, please repond in a funny way");
        match openai.chat_completion(&chat_id, &question, &co).await {
            Ok(r) => send_message_to_channel("ik8", "ch_mid", r.choice).await,
            Err(_e) => {}
        }
    }
}

async fn test_scraper_integration(url_inp: &str) -> Option<String> {
    if let Ok(res) = get_page_text(url_inp).await {
        send_message_to_channel("ik8", "ch_mid", res.to_string());

        return Some(res.to_string());
    }

    None
}

async fn test_chat(text_inp: String) {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let chat_id = format!("chat #N");
    let system = &format!("As a chatbot");

    let co = ChatOptions {
        model: ChatModel::GPT35Turbo,
        restart: true,
        system_prompt: Some(system),
    };

    let question = format!("Given the input: {text_inp}, provide a funny reply.");

    match openai.chat_completion(&chat_id, &question, &co).await {
        Ok(r) => {
            send_message_to_channel("ik8", "general", r.choice.clone());
        }
        Err(_e) => {}
    }
}

async fn test_openai_integration_summary(text_inp: String) {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let bpe = cl100k_base().unwrap();

    let feed_tokens_map = bpe.encode_ordinary(&text_inp);
    let chat_id = format!("news summary N");
    let system = &format!("As a news reporter AI,");

    let co = ChatOptions {
        model: ChatModel::GPT35Turbo,
        restart: true,
        system_prompt: Some(system),
    };

    for token_chunk in feed_tokens_map.chunks(2000) {
        let news_body = bpe.decode(token_chunk.to_vec()).unwrap();

        let question = format!(
            "Given a chunk of a new body text: {news_body}, please give a segment summary."
        );

        match openai.chat_completion(&chat_id, &question, &co).await {
            Ok(r) => {
                send_message_to_channel("ik8", "general", r.choice.clone());
            }
            Err(_e) => {}
        }
    }
}

#[tokio::main(flavor = "current_thread")]
pub async fn custom_gpt(sys_prompt: &str, user_prompt: &str) -> Option<String> {
    let system_prompt = serde_json::json!(
        {"role": "system", "content": sys_prompt}
    );
    let user_prompt = serde_json::json!(
        {"role": "user", "content": user_prompt}
    );

    if let Ok((res, _)) = chat(vec![system_prompt, user_prompt]).await {
        return Some(res);
    }
    None
}

pub async fn chat(message_obj: Vec<Value>) -> Result<(String, String), anyhow::Error> {
    dotenv().ok();
    let api_token = env::var("OPENAI_API_TOKEN")?;

    let params = serde_json::json!({
      "model": "gpt-3.5-turbo",
      "messages": message_obj,
      "temperature": 0.7,
      "top_p": 1,
      "n": 1,
      "stream": false,
      "max_tokens": 256,
      "presence_penalty": 0,
      "frequency_penalty": 0,
      "stop": "\n"
    });

    let uri = "https://api.openai.com/v1/chat/completions";

    let uri = Uri::try_from(uri)?;
    let mut writer = Vec::new();
    let body = serde_json::to_vec(&params)?;

    let bearer_token = format!("Bearer {}", api_token);
    let _response = Request::new(&uri)
        .method(Method::POST)
        .header("Authorization", &bearer_token)
        .header("Content-Type", "application/json")
        .header("Content-Length", &body.len())
        .body(&body)
        .send(&mut writer)?;

    // println!("Raw JSON: {}", String::from_utf8_lossy(&writer));
    let res = serde_json::from_slice::<ChatResponse>(&writer)?;
    let finish_reason = res.choices[0].finish_reason.clone();
    Ok((res.choices[0].message.content.to_string(), finish_reason))
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
}

#[derive(Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}
