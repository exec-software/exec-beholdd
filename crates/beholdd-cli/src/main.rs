use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    match beholdd_cli::run_from_env().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("beholdd: {error:#}");
            ExitCode::FAILURE
        }
    }
}
