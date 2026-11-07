use aes::cipher::block_padding::{Error as PaddingError, Padding};

/// Pad block with bytes with value equal to the number of bytes added.
///
/// PKCS#5 is described in [RFC 2898](https://tools.ietf.org/html/rfc2898).
#[derive(Clone, Copy, Debug)]
pub struct Pkcs5;

impl Pkcs5 {
    #[inline]
    fn unpad(block: &[u8], strict: bool) -> Result<&[u8], PaddingError> {
        // TODO: use bounds to check it at compile time
        if block.len() > 16 {
            panic!("block size is too big for PKCS#5");
        }
        let bs = block.len();
        let n = block[bs - 1];
        if n == 0 || n as usize > bs {
            return Err(PaddingError);
        }
        let s = bs - n as usize;
        if strict && block[s..bs - 1].iter().any(|&v| v != n) {
            return Err(PaddingError);
        }
        Ok(&block[..s])
    }
}

impl Padding for Pkcs5 {
    #[inline]
    fn raw_pad(block: &mut [u8], pos: usize) {
        // TODO: use bounds to check it at compile time for Padding<B>
        if block.len() > 16 {
            panic!("block size is too big for PKCS#5");
        }
        if pos >= block.len() {
            panic!("`pos` is bigger or equal to block size");
        }
        let n = (block.len() - pos) as u8;
        for b in &mut block[pos..] {
            *b = n;
        }
    }

    #[inline]
    fn raw_unpad(block: &[u8]) -> Result<&[u8], PaddingError> {
        Pkcs5::unpad(block, true)
    }
}
