use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use model::Model;
use needle::Needle;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};

const INDEX: &str = include_str!("../assets/index.html");
const SCRIPT: &str = include_str!("../assets/app.js");
const STYLE: &str = include_str!("../assets/style.css");
const FINETUNE_MESSAGE: &str = "fine-tuning runs in the Python tooling; load the resulting .cact here";

pub struct Config {
    pub host: String,
    pub port: u16,
    pub model: PathBuf,
    pub depth: Option<usize>,
}

struct Session {
    name: String,
    model: Arc<Model>,
    tools_json: String,
    agent: Needle,
}

pub struct Playground {
    server: Server,
    session: Session,
    uploads: PathBuf,
}

#[derive(Debug)]
pub struct PlaygroundError(pub String);

impl std::fmt::Display for PlaygroundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PlaygroundError {}

fn agent_for(model: &Arc<Model>, tools_json: &str) -> Needle {
    let tools = serde_json::from_str::<Value>(tools_json).ok().and_then(|value| value.as_array().cloned()).unwrap_or_default();
    Needle::builder().model(model).tools(tools).build()
}

fn name_of(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}

impl Playground {
    pub fn bind(config: &Config) -> Result<Playground, PlaygroundError> {
        let mut model = Model::open(&config.model).map_err(|error| PlaygroundError(error.to_string()))?;
        if let Some(depth) = config.depth {
            model = model.with_depth(depth).map_err(|error| PlaygroundError(error.to_string()))?;
        }
        let server = Server::http((config.host.as_str(), config.port)).map_err(|error| PlaygroundError(error.to_string()))?;
        let uploads = std::env::temp_dir().join(format!("rustyneedle-playground-{}", std::process::id()));
        fs::create_dir_all(&uploads).map_err(|error| PlaygroundError(error.to_string()))?;
        let agent = agent_for(&model, "[]");
        Ok(Playground { server, session: Session { name: name_of(&config.model), model, tools_json: "[]".into(), agent }, uploads })
    }

    pub fn address(&self) -> Option<SocketAddr> {
        self.server.server_addr().to_ip()
    }

    pub fn name(&self) -> &str {
        &self.session.name
    }

    pub fn run(mut self) {
        while let Ok(request) = self.server.recv() {
            self.handle(request);
        }
    }

    pub fn run_until(mut self, stop: std::sync::mpsc::Receiver<()>) {
        loop {
            if stop.try_recv().is_ok() {
                break;
            }
            match self.server.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(Some(request)) => self.handle(request),
                Ok(None) => {}
                Err(_) => break,
            }
        }
    }

    fn handle(&mut self, mut request: Request) {
        let path = request.url().split('?').next().unwrap_or("").to_owned();
        let method = request.method().clone();
        let outcome = match (&method, path.as_str()) {
            (Method::Get, "/" | "/index.html") => Reply::text(INDEX, "text/html; charset=utf-8"),
            (Method::Get, "/app.js") => Reply::text(SCRIPT, "application/javascript; charset=utf-8"),
            (Method::Get, "/style.css") => Reply::text(STYLE, "text/css; charset=utf-8"),
            (Method::Get, "/model") => Reply::json(json!({"name": self.session.name})),
            (Method::Get, "/finetune/status") => Reply::json(json!({"running": false, "step": "", "log": [], "checkpoint": null, "error": null})),
            (Method::Get, other) if other.starts_with("/download/") => self.download(&other["/download/".len()..]),
            (Method::Post, "/complete") => self.complete(&mut request),
            (Method::Post, "/reset") => {
                self.session.agent.reset();
                Reply::json(json!({"ok": true}))
            }
            (Method::Post, "/load-model") => self.load_model(&mut request),
            (Method::Post, "/finetune") => Reply::json(json!({"error": FINETUNE_MESSAGE})),
            _ => Reply::status(404, "not found"),
        };
        let _ = request.respond(outcome.into_response());
    }

    fn body(request: &mut Request) -> Vec<u8> {
        let mut bytes = Vec::new();
        let _ = request.as_reader().read_to_end(&mut bytes);
        bytes
    }

    fn complete(&mut self, request: &mut Request) -> Reply {
        let body: Value = serde_json::from_slice(&Self::body(request)).unwrap_or(Value::Null);
        let tools_json = match body.get("tools") {
            Some(Value::String(text)) => text.clone(),
            Some(other) => other.to_string(),
            None => "[]".to_owned(),
        };
        let query = body.get("query").and_then(Value::as_str).unwrap_or("");
        if tools_json != self.session.tools_json {
            self.session.agent = agent_for(&self.session.model, &tools_json);
            self.session.tools_json = tools_json;
        } else {
            self.session.agent.reset();
        }
        Reply::json(self.session.agent.complete(query).to_json())
    }

    fn load_model(&mut self, request: &mut Request) -> Reply {
        let name = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("X-Filename"))
            .map(|header| header.value.as_str().to_owned())
            .unwrap_or_else(|| "model.cact".into());
        let name = name_of(Path::new(&name));
        let destination = self.uploads.join(&name);
        if let Err(error) = fs::write(&destination, Self::body(request)) {
            return Reply::json(json!({"error": error.to_string()}));
        }
        match Model::open(&destination) {
            Ok(model) => {
                self.session.agent = agent_for(&model, "[]");
                self.session.model = model;
                self.session.tools_json = "[]".into();
                self.session.name = name;
                Reply::json(json!({"name": self.session.name}))
            }
            Err(error) => Reply::json(json!({"error": error.to_string()})),
        }
    }

    fn download(&self, name: &str) -> Reply {
        let file = self.uploads.join(name_of(Path::new(name)));
        match fs::read(&file) {
            Ok(bytes) => Reply { status: 200, body: bytes, content_type: "application/octet-stream".into() },
            Err(_) => Reply::status(404, "not found"),
        }
    }
}

struct Reply {
    status: u16,
    body: Vec<u8>,
    content_type: String,
}

impl Reply {
    fn text(body: &str, content_type: &str) -> Reply {
        Reply { status: 200, body: body.as_bytes().to_vec(), content_type: content_type.into() }
    }

    fn json(value: Value) -> Reply {
        Reply { status: 200, body: value.to_string().into_bytes(), content_type: "application/json".into() }
    }

    fn status(status: u16, message: &str) -> Reply {
        Reply { status, body: message.as_bytes().to_vec(), content_type: "text/plain".into() }
    }

    fn into_response(self) -> Response<std::io::Cursor<Vec<u8>>> {
        let header = Header::from_bytes(&b"Content-Type"[..], self.content_type.as_bytes()).expect("valid header");
        Response::from_data(self.body).with_status_code(self.status).with_header(header)
    }
}
