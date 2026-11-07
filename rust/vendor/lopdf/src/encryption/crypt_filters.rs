use super::DecryptionError;
use super::pkcs5::Pkcs5;
use super::rc4::Rc4;
use crate::ObjectId;
use aes::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit};
use md5::{Digest as _, Md5};
use rand::RngExt as _;

type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;
type Aes256CbcEnc = cbc::Encryptor<aes::Aes256>;

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;
type Aes256CbcDec = cbc::Decryptor<aes::Aes256>;

pub trait CryptFilter: std::fmt::Debug + Send + Sync {
    fn method(&self) -> &[u8];
    fn compute_key(&self, key: &[u8], obj_id: ObjectId) -> Result<Vec<u8>, DecryptionError>;
    fn encrypt(&self, key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, DecryptionError>;
    fn decrypt(&self, key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, DecryptionError>;
}

#[derive(Clone, Copy, Debug)]
pub struct IdentityCryptFilter;

impl CryptFilter for IdentityCryptFilter {
    fn method(&self) -> &[u8] {
        b"Identity"
    }

    fn compute_key(&self, key: &[u8], _obj_id: ObjectId) -> Result<Vec<u8>, DecryptionError> {
        Ok(key.to_vec())
    }

    fn encrypt(&self, _key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        Ok(plaintext.to_vec())
    }

    fn decrypt(&self, _key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        Ok(ciphertext.to_vec())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rc4CryptFilter;

impl CryptFilter for Rc4CryptFilter {
    fn method(&self) -> &[u8] {
        b"V2"
    }

    fn compute_key(&self, key: &[u8], obj_id: ObjectId) -> Result<Vec<u8>, DecryptionError> {
        let mut hasher = Md5::new();

        hasher.update(key);

        // For all strings and streams without crypt filter specifier; treating the object number
        // and generation number as binary integers, extend the original n-byte file encryption key
        // to n + 5 bytes by appending the low-order 3 bytes of the object number and the low-order
        // 2 bytes of the generation number in that order, low-order byte first.
        hasher.update(&obj_id.0.to_le_bytes()[..3]);
        hasher.update(&obj_id.1.to_le_bytes()[..2]);

        // Initialise the MD5 hash function and pass the result of the previous step as an input to
        // this function.
        //
        // Use the first (n + 5) bytes, up to a maximum of 16, of the output from the MD5 hash as
        // the key for the AES symmetric key algorithm.
        let key_len = std::cmp::min(key.len() + 5, 16);
        let key = hasher.finalize()[..key_len].to_vec();

        Ok(key)
    }

    fn encrypt(&self, key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        Ok(Rc4::new(key).encrypt(plaintext))
    }

    fn decrypt(&self, key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        Ok(Rc4::new(key).decrypt(ciphertext))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aes128CryptFilter;

impl CryptFilter for Aes128CryptFilter {
    fn method(&self) -> &[u8] {
        b"AESV2"
    }

    fn compute_key(&self, key: &[u8], obj_id: ObjectId) -> Result<Vec<u8>, DecryptionError> {
        let mut builder = Vec::with_capacity(key.len() + 9);

        builder.extend_from_slice(key);

        // For all strings and streams without crypt filter specifier; treating the object number
        // and generation number as binary integers, extend the original n-byte file encryption key
        // to n + 5 bytes by appending the low-order 3 bytes of the object number and the low-order
        // 2 bytes of the generation number in that order, low-order byte first.
        builder.extend_from_slice(&obj_id.0.to_le_bytes()[..3]);
        builder.extend_from_slice(&obj_id.1.to_le_bytes()[..2]);

        // If using the AES algorithm, extend the file encryption key an additional 4 bytes by
        // adding the value "sAlT".
        builder.extend_from_slice(b"sAlT");

        // Initialise the MD5 hash function and pass the result of the previous step as an input to
        // this function.
        //
        // Use the first (n + 5) bytes, up to a maximum of 16, of the output from the MD5 hash as
        // the key for the AES symmetric key algorithm.
        let key_len = std::cmp::min(key.len() + 5, 16);
        let key = Md5::digest(builder)[..key_len].to_vec();

        Ok(key)
    }

    fn encrypt(&self, key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        // Ensure that the key is 128 bits (i.e., 16 bytes).
        if key.len() != 16 {
            return Err(DecryptionError::InvalidKeyLength);
        }
        let key: &[u8; 16] = key.try_into().map_err(|_| DecryptionError::InvalidKeyLength)?;

        // The ciphertext needs to be a multiple of 16 bytes to include the padding.
        let ciphertext_len = (plaintext.len() + 16) / 16 * 16;

        // Allocate sufficient bytes for the initialization vector, the ciphertext and the padding
        // combined.
        let mut ciphertext = Vec::with_capacity(16 + ciphertext_len);

        // Generate random numbers to populate the initialization vector.
        let mut rng = rand::rng();
        let mut iv = [0u8; 16];
        rng.fill(&mut iv);

        // Combine the IV and the plaintext.
        ciphertext.extend_from_slice(&iv);
        ciphertext.extend_from_slice(plaintext);
        ciphertext.resize(16 + ciphertext_len, 0);

        // Use the 128-bit AES-CBC algorithm with PKCS#5 padding to encrypt the plaintext.
        //
        // Strings and streams encrypted with AES shall use a padding scheme that is described in
        // the Internet RFC 2898, PKCS #5: Password-Based Cryptography Specification Version 2.0;
        // see the Bibliography. For an original message length of M, the pad shall consist of 16 -
        // (M mod 16) bytes whose value shall also be 16 - (M mod 16).
        Aes128CbcEnc::new(key.into(), &iv.into())
            .encrypt_padded::<Pkcs5>(&mut ciphertext[16..], plaintext.len())
            // Padding errors should not occur when encrypting, but avoid causing a panic.
            .map_err(|_| DecryptionError::Padding)?;

        Ok(ciphertext)
    }

    fn decrypt(&self, key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        // Ensure that the key is 128 bits (i.e., 16 bytes).
        if key.len() != 16 {
            return Err(DecryptionError::InvalidKeyLength);
        }
        let key: &[u8; 16] = key.try_into().map_err(|_| DecryptionError::InvalidKeyLength)?;

        // Ensure that the ciphertext length is a multiple of 16 bytes.
        if !ciphertext.len().is_multiple_of(16) {
            return Err(DecryptionError::InvalidCipherTextLength);
        }

        // There is nothing to decrypt if the ciphertext is empty or only contains the IV.
        if ciphertext.is_empty() || ciphertext.len() == 16 {
            return Ok(vec![]);
        }

        let mut iv = [0x00u8; 16];
        iv.copy_from_slice(&ciphertext[..16]);

        // Use the 128-bit AES-CBC algorithm with PKCS#5 padding to decrypt the ciphertext.
        //
        // Strings and streams encrypted with AES shall use a padding scheme that is described in
        // the Internet RFC 2898, PKCS #5: Password-Based Cryptography Specification Version 2.0;
        // see the Bibliography. For an original message length of M, the pad shall consist of 16 -
        // (M mod 16) bytes whose value shall also be 16 - (M mod 16).
        let data = &mut ciphertext[16..].to_vec();

        Ok(Aes128CbcDec::new(key.into(), &iv.into())
            .decrypt_padded::<Pkcs5>(data)
            .map_err(|_| DecryptionError::Padding)?
            .to_vec())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aes256CryptFilter;

impl CryptFilter for Aes256CryptFilter {
    fn method(&self) -> &[u8] {
        b"AESV3"
    }

    fn compute_key(&self, key: &[u8], _obj_id: ObjectId) -> Result<Vec<u8>, DecryptionError> {
        // Use the 32-byte file encryption key for the AES-256 symmetric key algorithm.
        Ok(key.to_vec())
    }

    fn encrypt(&self, key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        // Ensure that the key is 256 bits (i.e., 32 bytes).
        if key.len() != 32 {
            return Err(DecryptionError::InvalidKeyLength);
        }
        let key: &[u8; 32] = key.try_into().map_err(|_| DecryptionError::InvalidKeyLength)?;

        // The ciphertext needs to be a multiple of 16 bytes to include the padding.
        let ciphertext_len = (plaintext.len() + 16) / 16 * 16;

        // Allocate sufficient bytes for the initialization vector, the ciphertext and the padding
        // combined.
        let mut ciphertext = Vec::with_capacity(16 + ciphertext_len);

        // Generate random numbers to populate the initialization vector.
        let mut rng = rand::rng();
        let mut iv = [0u8; 16];
        rng.fill(&mut iv);

        // Combine the IV and the plaintext.
        ciphertext.extend_from_slice(&iv);
        ciphertext.extend_from_slice(plaintext);
        ciphertext.resize(16 + ciphertext_len, 0);

        // Use the 256-bit AES-CBC algorithm with PKCS#5 padding to encrypt the plaintext.
        //
        // Strings and streams encrypted with AES shall use a padding scheme that is described in
        // the Internet RFC 2898, PKCS #5: Password-Based Cryptography Specification Version 2.0;
        // see the Bibliography. For an original message length of M, the pad shall consist of 16 -
        // (M mod 16) bytes whose value shall also be 16 - (M mod 16).
        Aes256CbcEnc::new(key.into(), &iv.into())
            .encrypt_padded::<Pkcs5>(&mut ciphertext[16..], plaintext.len())
            // Padding errors should not occur when encrypting, but avoid causing a panic.
            .map_err(|_| DecryptionError::Padding)?;

        Ok(ciphertext)
    }

    fn decrypt(&self, key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, DecryptionError> {
        // Ensure that the key is 256 bits (i.e., 32 bytes).
        if key.len() != 32 {
            return Err(DecryptionError::InvalidKeyLength);
        }
        let key: &[u8; 32] = key.try_into().map_err(|_| DecryptionError::InvalidKeyLength)?;

        // Ensure that the ciphertext length is a multiple of 16 bytes.
        if !ciphertext.len().is_multiple_of(16) {
            return Err(DecryptionError::InvalidCipherTextLength);
        }

        // There is nothing to decrypt if the ciphertext is empty or only contains the IV.
        if ciphertext.is_empty() || ciphertext.len() == 16 {
            return Ok(vec![]);
        }

        let mut iv = [0x00u8; 16];
        iv.copy_from_slice(&ciphertext[..16]);

        // Use the 256-bit AES-CBC algorithm with PKCS#7 padding to decrypt the ciphertext.
        //
        // Strings and streams encrypted with AES shall use a padding scheme that is described in
        // the Internet RFC 2898, PKCS #5: Password-Based Cryptography Specification Version 2.0;
        // see the Bibliography. For an original message length of M, the pad shall consist of 16 -
        // (M mod 16) bytes whose value shall also be 16 - (M mod 16).
        let data = &mut ciphertext[16..].to_vec();

        Ok(Aes256CbcDec::new(key.into(), &iv.into())
            .decrypt_padded::<Pkcs5>(data)
            .map_err(|_| DecryptionError::Padding)?
            .to_vec())
    }
}
