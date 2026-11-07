//! Whole-write retry policy, independent of Python and PDF constructors.

pub trait WriteAttempt {
    type Value;
    type Error;
    fn attempt(&mut self, check_font_exists: bool)
    -> std::result::Result<Self::Value, Self::Error>;
    fn close(&mut self) -> std::result::Result<(), Self::Error>;
    fn ordinary(&self, error: &Self::Error) -> bool;
    fn failed(&mut self, error: &Self::Error);
    fn recovered(&mut self);
}

/// Cleanup runs after every attempt, including cancellation. A cleanup bridge
/// failure cannot be turned into an ordinary retry by an earlier body error.
pub fn write_with_retry<A: WriteAttempt + ?Sized>(
    run: &mut A,
) -> std::result::Result<(A::Value, u8), A::Error> {
    for number in 1..=2 {
        let body = run.attempt(number == 2);
        let cleanup = run.close();
        let result = match (body, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(error),
            (Err(primary), Err(cleanup)) if run.ordinary(&primary) && !run.ordinary(&cleanup) => {
                Err(cleanup)
            }
            (Err(primary), _) => Err(primary),
        };
        match result {
            Ok(value) => return Ok((value, number)),
            Err(error) => {
                if !run.ordinary(&error) {
                    return Err(error);
                }
                run.failed(&error);
                if number == 2 {
                    return Err(error);
                }
                run.recovered();
            }
        }
    }
    unreachable!("at most two attempts")
}

#[cfg(test)]
mod retry_tests;
