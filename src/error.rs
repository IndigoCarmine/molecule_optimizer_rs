use thiserror::Error;

/// Crate-wide error type.
#[derive(Debug, Error)]
pub enum Error {
    #[error("parse error (line {line}): {msg}")]
    Parse { line: usize, msg: String },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unknown element: {0}")]
    UnknownElement(String),
    #[error("perception error: {0}")]
    Perception(String),
    #[error("force field setup error: {0}")]
    ForceField(String),
    #[error("missing force field parameters: {0}")]
    MissingParameters(String),
    #[error("embedding failed: {0}")]
    Embedding(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("job cancelled")]
    Cancelled,
    #[error("job failed: {0}")]
    Job(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn parse(line: usize, msg: impl Into<String>) -> Self {
        Error::Parse { line, msg: msg.into() }
    }
}
