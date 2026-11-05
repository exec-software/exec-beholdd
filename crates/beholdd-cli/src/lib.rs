//! Argument parsing and process wiring for the `beholdd` binary.
//!
//! Model loading and inference remain in `beholdd-runtime`; HTTP remains in
//! `beholdd-server`. This crate deliberately contains only the small amount of policy
//! needed to turn process arguments into those two layers.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use beholdd_runtime::{InferenceRuntime, NativeRuntime, RuntimeConfig};
use tokio::net::TcpListener;

/// Default loopback interface used by `beholdd serve`.
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Default v2 port used by `beholdd serve`.
pub const DEFAULT_PORT: u16 = 6768;

/// The command selected on the command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Run the OpenAI-compatible HTTP server.
    Serve(ServeArgs),
}

/// Arguments accepted by the `serve` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    /// Runtime configuration file.
    pub config: PathBuf,
    /// Optional server host override.
    pub host: Option<String>,
    /// Optional server port override.
    pub port: Option<u16>,
}

/// A command-line parsing or environment error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    /// No command was provided.
    MissingCommand,
    /// The command is not supported.
    UnknownCommand(OsString),
    /// An option is not supported for the selected command.
    UnknownArgument(OsString),
    /// An option appeared more than once.
    DuplicateArgument(&'static str),
    /// An option was not followed by a value.
    MissingValue(&'static str),
    /// An option value was not valid Unicode.
    NonUnicodeValue(&'static str),
    /// A host override was empty.
    EmptyHost,
    /// A port override was not an integer from 1 through 65535.
    InvalidPort(OsString),
    /// A configured home environment variable was present but empty.
    EmptyEnvironment(&'static str),
    /// Neither `BEHOLDD_HOME` nor `HOME` could provide a default configuration path.
    MissingHome,
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCommand => formatter.write_str("missing command (expected `serve`)"),
            Self::UnknownCommand(command) => {
                write!(formatter, "unknown command `{}`", command.to_string_lossy())
            }
            Self::UnknownArgument(argument) => {
                write!(
                    formatter,
                    "unknown argument `{}`",
                    argument.to_string_lossy()
                )
            }
            Self::DuplicateArgument(argument) => {
                write!(
                    formatter,
                    "argument `{argument}` may only be specified once"
                )
            }
            Self::MissingValue(argument) => {
                write!(formatter, "argument `{argument}` requires a value")
            }
            Self::NonUnicodeValue(argument) => {
                write!(formatter, "value for `{argument}` is not valid Unicode")
            }
            Self::EmptyHost => formatter.write_str("value for `--host` may not be empty"),
            Self::InvalidPort(port) => write!(
                formatter,
                "invalid port `{}` (expected an integer from 1 through 65535)",
                port.to_string_lossy()
            ),
            Self::EmptyEnvironment(name) => write!(formatter, "{name} is set but empty"),
            Self::MissingHome => formatter.write_str(
                "cannot find default config: set BEHOLDD_HOME or HOME, or pass --config",
            ),
        }
    }
}

impl std::error::Error for CliError {}

/// Parse process arguments, including the program name in the first position.
pub fn parse_args<I, S>(arguments: I) -> Result<Command, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    parse_args_with_env(arguments, |name| env::var_os(name))
}

/// Parse process arguments with an injectable environment lookup.
///
/// This is public so embedders and distribution wrappers can apply the exact CLI policy
/// without mutating the process environment.
pub fn parse_args_with_env<I, S, F>(arguments: I, env_var: F) -> Result<Command, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
    F: Fn(&str) -> Option<OsString>,
{
    let mut arguments = arguments.into_iter().map(Into::into);
    let _program = arguments.next();
    let Some(command) = arguments.next() else {
        return Err(CliError::MissingCommand);
    };
    if command != OsStr::new("serve") {
        return Err(CliError::UnknownCommand(command));
    }

    let mut config = None;
    let mut host = None;
    let mut port = None;
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--config") => {
                ensure_unset(&config, "--config")?;
                config = Some(option_value(&mut arguments, "--config")?);
            }
            Some("--host") => {
                ensure_unset(&host, "--host")?;
                let value = option_value(&mut arguments, "--host")?;
                let value = value
                    .into_string()
                    .map_err(|_| CliError::NonUnicodeValue("--host"))?;
                if value.is_empty() {
                    return Err(CliError::EmptyHost);
                }
                host = Some(value);
            }
            Some("--port") => {
                ensure_unset(&port, "--port")?;
                let value = option_value(&mut arguments, "--port")?;
                let parsed = value
                    .to_str()
                    .and_then(|value| value.parse::<u16>().ok())
                    .filter(|port| *port != 0)
                    .ok_or_else(|| CliError::InvalidPort(value.clone()))?;
                port = Some(parsed);
            }
            _ => return Err(CliError::UnknownArgument(argument)),
        }
    }

    let config = match config {
        Some(path) => PathBuf::from(path),
        None => default_config_path_with(&env_var)?,
    };
    Ok(Command::Serve(ServeArgs { config, host, port }))
}

/// Return the default `config.yaml` path using the current process environment.
pub fn default_config_path() -> Result<PathBuf, CliError> {
    default_config_path_with(&|name| env::var_os(name))
}

/// Parse the current process arguments and run the selected command.
pub async fn run_from_env() -> anyhow::Result<()> {
    let command = parse_args(env::args_os()).context("invalid command line")?;
    run(command).await
}

/// Run a previously parsed command.
pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Serve(arguments) => serve(arguments).await,
    }
}

async fn serve(arguments: ServeArgs) -> anyhow::Result<()> {
    let config = RuntimeConfig::from_path(&arguments.config).with_context(|| {
        format!(
            "failed to read runtime configuration {}",
            arguments.config.display()
        )
    })?;
    let runtime: Arc<dyn InferenceRuntime> =
        Arc::new(NativeRuntime::new(config).context("failed to initialize the native runtime")?);
    let host = overridden_host(DEFAULT_HOST, &arguments.host);
    let port = overridden_port(DEFAULT_PORT, arguments.port);
    let listener = TcpListener::bind((host, port))
        .await
        .with_context(|| format!("failed to bind {host}:{port}"))?;
    let address = listener
        .local_addr()
        .context("failed to inspect the server listener")?;
    let shutdown_signal = shutdown_signal()?;
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let serving = beholdd_server::serve_with_shutdown(listener, runtime, async {
        let _ = stopped.await;
    });
    tokio::pin!(serving);
    eprintln!("beholdd v2 listening on http://{address}");
    tokio::select! {
        result = &mut serving => result,
        result = shutdown_signal => {
            result?;
            let _ = stop.send(());
            serving.await
        }
    }
}

// Register Unix handlers before announcing readiness. Graceful shutdown stops
// admission and drains HTTP connections; active inference still owns its normal
// runtime admission and resources until synchronous execution has completed.
#[cfg(unix)]
fn shutdown_signal() -> anyhow::Result<impl std::future::Future<Output = anyhow::Result<()>>> {
    use tokio::signal::unix::{signal, SignalKind};

    let mut interrupt = signal(SignalKind::interrupt()).context("registering SIGINT handler")?;
    let mut terminate = signal(SignalKind::terminate()).context("registering SIGTERM handler")?;
    Ok(async move {
        tokio::select! {
            _ = interrupt.recv() => {},
            _ = terminate.recv() => {},
        }
        Ok(())
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> anyhow::Result<impl std::future::Future<Output = anyhow::Result<()>>> {
    Ok(async {
        tokio::signal::ctrl_c()
            .await
            .context("waiting for the shutdown signal")
    })
}

fn default_config_path_with<F>(env_var: &F) -> Result<PathBuf, CliError>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(home) = env_var("BEHOLDD_HOME") {
        if home.is_empty() {
            return Err(CliError::EmptyEnvironment("BEHOLDD_HOME"));
        }
        return Ok(PathBuf::from(home).join("config.yaml"));
    }
    let home = env_var("HOME").ok_or(CliError::MissingHome)?;
    if home.is_empty() {
        return Err(CliError::EmptyEnvironment("HOME"));
    }
    Ok(PathBuf::from(home).join(".beholdd").join("config.yaml"))
}

fn option_value<I>(arguments: &mut I, option: &'static str) -> Result<OsString, CliError>
where
    I: Iterator<Item = OsString>,
{
    let value = arguments.next().ok_or(CliError::MissingValue(option))?;
    if value.to_string_lossy().starts_with("--") {
        return Err(CliError::MissingValue(option));
    }
    Ok(value)
}

fn ensure_unset<T>(slot: &Option<T>, name: &'static str) -> Result<(), CliError> {
    if slot.is_some() {
        return Err(CliError::DuplicateArgument(name));
    }
    Ok(())
}

/// Apply a host override if one was supplied.
pub fn overridden_host<'a>(configured: &'a str, override_host: &'a Option<String>) -> &'a str {
    override_host.as_deref().unwrap_or(configured)
}

/// Apply a port override if one was supplied.
pub fn overridden_port(configured: u16, override_port: Option<u16>) -> u16 {
    override_port.unwrap_or(configured)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<OsString> {
        None
    }

    #[test]
    fn parses_explicit_serve_arguments() {
        let command = parse_args_with_env(
            [
                "beholdd",
                "serve",
                "--config",
                "/tmp/config.yaml",
                "--host",
                "127.0.0.1",
                "--port",
                "6768",
            ],
            no_env,
        )
        .unwrap();
        assert_eq!(
            command,
            Command::Serve(ServeArgs {
                config: PathBuf::from("/tmp/config.yaml"),
                host: Some("127.0.0.1".into()),
                port: Some(6768),
            })
        );
    }

    #[test]
    fn uses_beholdd_home_before_home() {
        let command = parse_args_with_env(["beholdd", "serve"], |name| match name {
            "BEHOLDD_HOME" => Some("/srv/beholdd".into()),
            "HOME" => Some("/home/ignored".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            command,
            Command::Serve(ServeArgs {
                config: PathBuf::from("/srv/beholdd/config.yaml"),
                host: None,
                port: None,
            })
        );
    }

    #[test]
    fn falls_back_to_dot_beholdd_under_home() {
        let command = parse_args_with_env(["beholdd", "serve"], |name| match name {
            "HOME" => Some("/home/tester".into()),
            _ => None,
        })
        .unwrap();
        let Command::Serve(arguments) = command;
        assert_eq!(
            arguments.config,
            PathBuf::from("/home/tester/.beholdd/config.yaml")
        );
    }

    #[test]
    fn rejects_empty_home_variables_like_the_runtime_loader() {
        assert_eq!(
            parse_args_with_env(["beholdd", "serve"], |name| match name {
                "BEHOLDD_HOME" => Some(OsString::new()),
                "HOME" => Some("/home/ignored".into()),
                _ => None,
            }),
            Err(CliError::EmptyEnvironment("BEHOLDD_HOME"))
        );
        assert_eq!(
            parse_args_with_env(["beholdd", "serve"], |name| match name {
                "HOME" => Some(OsString::new()),
                _ => None,
            }),
            Err(CliError::EmptyEnvironment("HOME"))
        );
    }

    #[test]
    fn rejects_duplicates_unknowns_and_bad_ports() {
        assert_eq!(
            parse_args_with_env(
                ["beholdd", "serve", "--config", "one", "--config", "two"],
                no_env
            ),
            Err(CliError::DuplicateArgument("--config"))
        );
        assert_eq!(
            parse_args_with_env(["beholdd", "serve", "--config", "one", "--wat"], no_env),
            Err(CliError::UnknownArgument("--wat".into()))
        );
        assert_eq!(
            parse_args_with_env(
                ["beholdd", "serve", "--config", "one", "--port", "0"],
                no_env
            ),
            Err(CliError::InvalidPort("0".into()))
        );
        assert_eq!(
            parse_args_with_env(
                [
                    "beholdd",
                    "serve",
                    "--config",
                    "one",
                    "--port",
                    "6768",
                    "--port",
                    "not-a-port"
                ],
                no_env
            ),
            Err(CliError::DuplicateArgument("--port"))
        );
    }

    #[test]
    fn rejects_an_option_where_a_value_is_required() {
        assert_eq!(
            parse_args_with_env(
                ["beholdd", "serve", "--config", "--host", "localhost"],
                no_env
            ),
            Err(CliError::MissingValue("--config"))
        );
    }

    #[test]
    fn explicit_config_does_not_require_a_home() {
        let command =
            parse_args_with_env(["beholdd", "serve", "--config", "relative.yaml"], no_env).unwrap();
        let Command::Serve(arguments) = command;
        assert_eq!(arguments.config, PathBuf::from("relative.yaml"));
    }

    #[test]
    fn applies_overrides_without_changing_configured_values() {
        assert_eq!(overridden_host("0.0.0.0", &None), "0.0.0.0");
        assert_eq!(
            overridden_host("0.0.0.0", &Some("127.0.0.1".into())),
            "127.0.0.1"
        );
        assert_eq!(overridden_port(6767, None), 6767);
        assert_eq!(overridden_port(6767, Some(6768)), 6768);
    }
}
