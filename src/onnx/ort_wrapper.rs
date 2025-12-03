#[cfg(not(feature = "cuda"))]
use ort::execution_providers::cpu::CPUExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::logging::LogLevel;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

pub trait OrtWrapper {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];

        #[cfg(not(feature = "cuda"))]
        let providers = [CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(builder) => {
                let session = builder
                    .with_execution_providers(providers)
                    .map_err(|e| format!("Failed to build session: {e}"))?
                    .with_log_level(LogLevel::Warning)
                    .map_err(|e| format!("Failed to set log level: {e}"))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {e}"))?;
                self.set_session(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {e}")),
        }
    }

    fn set_session(&mut self, session: Session);
    fn get_session(&self) -> Option<&Session>;
}
