use super::DecryptionError;
use super::rc4::Rc4;
use crate::encodings;
use crate::encryption::Permissions;
use crate::{Document, Error, Object};
use aes::cipher::{BlockModeDecrypt as _, BlockModeEncrypt as _, KeyInit as _, KeyIvInit as _};
use md5::{Digest as _, Md5};
use rand::RngExt as _;
use sha2::{Sha256, Sha384, Sha512};

type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;
type Aes256CbcEnc = cbc::Encryptor<aes::Aes256>;
type Aes256EbcEnc = ecb::Encryptor<aes::Aes256>;

type Aes256CbcDec = cbc::Decryptor<aes::Aes256>;
type Aes256EbcDec = ecb::Decryptor<aes::Aes256>;
type AesBlock = aes::cipher::Block<aes::Aes128>;

fn aes_block_mut(block: &mut [u8]) -> &mut AesBlock {
    block.try_into().expect("AES block must be 16 bytes")
}

// If the password string is less than 32 bytes long, pad it by appending the required number of
// additional bytes from the beginning of the following padding string.
const PAD_BYTES: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08, 0x2E, 0x2E, 0x00,
    0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

#[derive(Clone, Debug, Default)]
pub struct PasswordAlgorithm {
    pub(crate) encrypt_metadata: bool,
    pub(crate) length: Option<usize>,
    pub(crate) version: i64,
    pub(crate) revision: i64,
    pub(crate) owner_value: Vec<u8>,
    pub(crate) owner_encrypted: Vec<u8>,
    pub(crate) user_value: Vec<u8>,
    pub(crate) user_encrypted: Vec<u8>,
    pub(crate) permissions: Permissions,
    pub(crate) permission_encrypted: Vec<u8>,
}

impl TryFrom<&Document> for PasswordAlgorithm {
    type Error = Error;

    fn try_from(value: &Document) -> Result<Self, Self::Error> {
        // Get the encrypted dictionary.
        let encrypted = value
            .get_encrypted()
            .map_err(|_| DecryptionError::MissingEncryptDictionary)?;

        // Get the EncryptMetadata field.
        let encrypt_metadata = encrypted
            .get(b"EncryptMetadata")
            .unwrap_or(&Object::Boolean(true))
            .as_bool()
            .map_err(|_| DecryptionError::InvalidType)?;

        // Get the Length field if any. Make sure that if it is present that it is a 64-bit integer and
        // that it can be converted to an unsigned size.
        let length: Option<usize> = if encrypted.get(b"Length").is_ok() {
            Some(encrypted.get(b"Length")?.as_i64()?.try_into()?)
        } else {
            None
        };

        // Get the V field.
        let version = encrypted
            .get(b"V")
            .map_err(|_| DecryptionError::MissingVersion)?
            .as_i64()
            .map_err(|_| DecryptionError::InvalidType)?;

        // A code specifying the algorithm to be used in encrypting and decrypting the document.
        match version {
            // (Deprecated in PDF 2.0) An algorithm that is undocumented. This value shall not be
            // used.
            0 => return Err(DecryptionError::InvalidVersion)?,
            // (PDF 1.4; deprecated in PDF 2.0) Indicates the use of encryption of data using the
            // RC4 or AES algorithms with a file encryption key length of 40 bits.
            1 => (),
            // (PDF 1.4; deprecated in PDF 2.0) Indicates the use of encryption of data using the
            // RC4 or AES algorithms but permitting file encryption key lengths greater or 40 bits.
            2 => (),
            // (PDF 1.4; deprecated in PDF 2.0) An unpublished algorithm that permits encryption
            // key lengths ranging from 40 to 128 bits. This value shall not appear in a conforming
            // PDF file.
            3 => return Err(DecryptionError::InvalidVersion)?,
            // (PDF 1.5; deprecated in PDF 2.0) The security handler defines the use of encryption
            // and decryption in the document, using the rules specified by the CF, StmF and StrF
            // entries using encryption of data using the RC4 or AES algorithms (deprecated in PDF
            // 2.0) with a file encryption key length of 128 bits.
            4 => (),
            // (PDF 2.0) The security handler defines the use of encryption and decryption in the
            // document, using the rules specified by the CF, StmF, StrF and EFF entries using
            // encryption of data using the AES algorithms with a file encryption key length of 256
            // bits.
            5 => (),
            // Unknown codes.
            _ => Err(DecryptionError::UnsupportedVersion)?,
        }

        // V4 and V5 fix the length of the file encryption key at 128 and 256 bits, so the Length
        // entry is optional for them and plenty of producers leave it out. Without a default here
        // the key derivation below falls back to 40 bits and the password check fails on files
        // that other readers open without complaint.
        let length = length.or(match version {
            4 => Some(128),
            5 => Some(256),
            _ => None,
        });

        // The length of the file encryption key shall only be present if V is 2 or 3 (but
        // documents with higher values for V seem to have this field).
        if let Some(length) = length {
            match version {
                // Although "Optional" and/or not required for V1 it appears in some documents
                // with a default value of 40.
                1 => {
                    if length != 40 {
                        Err(DecryptionError::InvalidKeyLength)?;
                    }
                }
                // The length of the file encryption key shall be a multiple of 8 in the range 40
                // to and including 128.
                2..=3 => {
                    if length % 8 != 0 || !(40..=128).contains(&length) {
                        Err(DecryptionError::InvalidKeyLength)?;
                    }
                }
                // The Length field should not be present if V is 4. However, if it is present it
                // must be 128.
                4 => {
                    if length != 128 {
                        Err(DecryptionError::InvalidKeyLength)?;
                    }
                }
                // The Length field should not be present if V is 5. However, if it is present it
                // must be 256.
                5 => {
                    if length != 256 {
                        Err(DecryptionError::InvalidKeyLength)?;
                    }
                }
                // The Length field may not be present otherwise.
                _ => Err(DecryptionError::InvalidKeyLength)?,
            }
        }

        // Get the R field.
        let revision = encrypted
            .get(b"R")
            .map_err(|_| DecryptionError::MissingRevision)?
            .as_i64()
            .map_err(|_| DecryptionError::InvalidType)?;

        // Get the owner value and owner encrypted blobs.
        let mut owner_value = encrypted
            .get(b"O")
            .map_err(|_| DecryptionError::MissingOwnerPassword)?
            .as_str()
            .map_err(|_| DecryptionError::InvalidType)?
            .to_vec();

        // The owner value is 32 bytes long if the value of R is 4 or less.
        if revision <= 4 && owner_value.len() != 32 {
            Err(DecryptionError::InvalidHashLength)?;
        }

        // The owner value is 48 bytes long if the value of R is 5 or greater.
        // Some PDF writers pad /O with trailing zeros beyond 48 bytes; truncate
        // to the spec-required length so these documents are accepted.
        if revision >= 5 {
            if owner_value.len() < 48 {
                Err(DecryptionError::InvalidHashLength)?;
            }
            owner_value.truncate(48);
        }

        let owner_encrypted = encrypted
            .get(b"OE")
            .and_then(Object::as_str)
            .map(|s| s.to_vec())
            .ok()
            .unwrap_or_default();

        // The owner encrypted blob is required if R is 5 or greater and the blob shall be 32 bytes
        // long.
        if revision >= 5 && owner_encrypted.len() != 32 {
            Err(DecryptionError::InvalidCipherTextLength)?;
        }

        // Get the user value and user encrypted blobs.
        let mut user_value = encrypted
            .get(b"U")
            .map_err(|_| DecryptionError::MissingUserPassword)?
            .as_str()
            .map_err(|_| DecryptionError::InvalidType)?
            .to_vec();

        // The user value is 32 bytes long if the value of R is 4 or less.
        if revision <= 4 && user_value.len() != 32 {
            Err(DecryptionError::InvalidHashLength)?;
        }

        // The user value is 48 bytes long if the value of R is 5 or greater.
        // Some PDF writers pad /U with trailing zeros beyond 48 bytes; truncate
        // to the spec-required length so these documents are accepted.
        if revision >= 5 {
            if user_value.len() < 48 {
                Err(DecryptionError::InvalidHashLength)?;
            }
            user_value.truncate(48);
        }

        let user_encrypted = encrypted
            .get(b"UE")
            .and_then(Object::as_str)
            .map(|s| s.to_vec())
            .ok()
            .unwrap_or_default();

        // The user encrypted blob is required if R is 5 or greater and the blob shall be 32 bytes
        // long.
        if revision >= 5 && user_encrypted.len() != 32 {
            Err(DecryptionError::InvalidCipherTextLength)?;
        }

        // Get the permission value and permission encrypted blobs.
        let permission_value = encrypted
            .get(b"P")
            .map_err(|_| DecryptionError::MissingPermissions)?
            .as_i64()
            .map_err(|_| DecryptionError::InvalidType)? as u64;

        let permissions = Permissions::from_bits_retain(permission_value);

        let permission_encrypted = encrypted
            .get(b"Perms")
            .and_then(Object::as_str)
            .map(|s| s.to_vec())
            .ok()
            .unwrap_or_default();

        // The permission encrypted blob is required if R is 65 or greater and the blob shall be
        // 16 bytes long.
        if revision >= 5 && permission_encrypted.len() != 16 {
            Err(DecryptionError::InvalidCipherTextLength)?;
        }

        Ok(Self {
            encrypt_metadata,
            length,
            version,
            revision,
            owner_value,
            owner_encrypted,
            user_value,
            user_encrypted,
            permissions,
            permission_encrypted,
        })
    }
}

impl PasswordAlgorithm {
    /// Sanitize the password (revision 4 and earlier).
    ///
    /// This implements the first step of Algorithm 2 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    pub(crate) fn sanitize_password_r4(&self, password: &str) -> Result<Vec<u8>, DecryptionError> {
        // The password string is generated from host system codepage characters (or system scripts) by
        // first converting the string to PDFDocEncoding. If the input is Unicode, first convert to a
        // codepage encoding, and then to PDFDocEncoding for backward compatibility.
        let password = encodings::string_to_bytes(&encodings::PDF_DOC_ENCODING, password);

        Ok(password)
    }

    /// Compute a file encryption key in order to encrypt/decrypt a document (revision 4 and
    /// earlier).
    ///
    /// This implements Algorithm 2 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    pub(crate) fn compute_file_encryption_key_r4<P>(
        &self, doc: &Document, password: P,
    ) -> Result<Vec<u8>, DecryptionError>
    where
        P: AsRef<[u8]>,
    {
        let password = password.as_ref();

        // Pad or truncate the resulting password string to exactly 32 bytes. If the password string is
        // more than 32 bytes long, use only its first 32 bytes; if it is less than 32 bytes long, pad
        // it by appending the required number of additional bytes from the beginning of the following
        // padding string (see `PAD_BYTES`).
        //
        // That is, if the password is n bytes long, append the first 32 - n bytes of the padding
        // string to the end of the password string. If the password string is empty (zero-length),
        // meaning there is no user password, substitute the entire padding string in its place.
        //
        // i.e., we will simply calculate `len = min(password length, 32)` and use the first len bytes
        // of password and the first len bytes of `PAD_BYTES`.
        let len = password.len().min(32);

        // Initialize the MD5 hash function and pass the result as input to this function.
        let mut hasher = Md5::new();

        hasher.update(&password[..len]);
        hasher.update(&PAD_BYTES[..32 - len]);

        // Pass the value of the encryption dictionary's O entry (owner password hash) to the MD5 hash
        // function.
        hasher.update(&self.owner_value);

        // Convert the integer value of the P entry (permissions) to a 32-bit unsigned binary number
        // and pass these bytes to the MD5 hash function, low-order byte first.
        //
        // We don't actually care about the permissions, but we need the correct value to derive the
        // correct key.
        hasher.update((self.permissions.bits() as u32).to_le_bytes());

        // Pass the first element of the file's file identifier array (the value of the ID entry in the
        // document's trailer dictionary to the MD5 hash function.
        let file_id_0 = doc
            .trailer
            .get(b"ID")
            .map_err(|_| DecryptionError::MissingFileID)?
            .as_array()
            .map_err(|_| DecryptionError::InvalidType)?
            .first()
            .ok_or(DecryptionError::InvalidType)?
            .as_str()
            .map_err(|_| DecryptionError::InvalidType)?;
        hasher.update(file_id_0);

        // (Security handlers of revision 4 or greater) If document metadata is not being encrypted,
        // pass 4 bytes with the value 0xFFFFFFFF to the MD5 hash function.
        if self.revision >= 4 && !self.encrypt_metadata {
            hasher.update(b"\xff\xff\xff\xff");
        }

        // Finish the hash.
        let mut hash = hasher.finalize();

        // (Security handlers of revision 3 or greater) Do the following 50 times: take the output from
        // the previous MD5 hash and pass the first n bytes of the output as input into a new MD5 hash,
        // where n is the number of bytes of the file encryption key as defined by the value of the
        // encryption dictionary's Length entry.
        let n = if self.revision >= 3 {
            self.length.unwrap_or(40) / 8
        } else {
            5
        };

        // The maximum supported key length is 16 bytes (128 bits) due to the use of MD5.
        if n > 16 {
            return Err(DecryptionError::InvalidKeyLength);
        }

        if self.revision >= 3 {
            for _ in 0..50 {
                hash = Md5::digest(&hash[..n]);
            }
        }

        // Set the file encryption key to the first n bytes of the output from the final MD5 hash,
        // where n shall always be 5 for security handlers of revision 2 but, for security handlers of
        // revision 3 or greater, shall depend on the value of the encrpytion dictionary's Length
        // entry.
        Ok(hash[..n].to_vec())
    }

    /// Sanitize the password (revision 6 and later).
    ///
    /// This implements the first step of Algorithm 2.A as described in ISO 32000-2:2020 (PDF 2.0).
    pub(crate) fn sanitize_password_r6(&self, password: &str) -> Result<Vec<u8>, DecryptionError> {
        // The UTF-8 password string shall be generated from Unicode input by processing the input
        // with the SASLprep (Internet RFC 4013) profile of stringprep (Internet RFC 3454) using
        // the Normalize and BiDi options, and then coverting to a UTF-8 representation.
        Ok(stringprep::saslprep(password)?.as_bytes().to_vec())
    }

    /// Compute a file encryption key in order to encrypt/decrypt a document (revision 6 and
    /// later).
    ///
    /// This implements Algorithm 2.A as described in ISO 32000-2:2020 (PDF 2.0).
    fn compute_file_encryption_key_r6<P>(&self, password: P) -> Result<Vec<u8>, DecryptionError>
    where
        P: AsRef<[u8]>,
    {
        let mut password = password.as_ref();

        let hashed_owner_password = &self.owner_value[0..][..32];
        let owner_validation_salt = &self.owner_value[32..][..8];
        let owner_key_salt = &self.owner_value[40..][..8];

        let hashed_user_password = &self.user_value[0..][..32];
        let user_validation_salt = &self.user_value[32..][..8];
        let user_key_salt = &self.user_value[40..][..8];

        // Truncate the UTF-8 representation to 127 bytes if it is longer than 127 bytes.
        if password.len() > 127 {
            password = &password[..127];
        }

        // Test the password against the owner key by computing a hash using algorithm 2.B with an
        // input string consisting of the UTF-8 password concatenated with the 8 bytes of owner
        // validation salt, concatenated with the 48-byte U string. If the 32-byte result matches
        // the first 32 bytes of the O string, this is the owner password.
        if self.compute_hash(password, owner_validation_salt, Some(&self.user_value))? == hashed_owner_password {
            // Compute an intermediate owner key by computing a hash using algorithm 2.B with an
            // input string consisting of the UTF-8 owner password concatenated with the 8 bytes of
            // owner key salt, concatenated with the 48-byte U string.
            let hash = self.compute_hash(password, owner_key_salt, Some(&self.user_value))?;

            let mut key = [0u8; 32];
            key.copy_from_slice(&hash);

            // The 32-byte result is the key used to decrypt the 32-byte OE string using AES-256 in
            // CBC mode with no padding and an initialization vector of zero. The 32-byte result is
            // the file encryption key.
            let iv = [0u8; 16];

            let mut owner_encrypted = self.owner_encrypted.clone();
            let mut decryptor = Aes256CbcDec::new(&key.into(), &iv.into());

            for block in owner_encrypted.as_chunks_mut::<16>().0 {
                decryptor.decrypt_block(aes_block_mut(block));
            }

            return Ok(owner_encrypted);
        }

        // Note: this step is not in the specification, but is a precaution.
        //
        // Test the password against the user key by computing a hash using algorithm 2.B with an
        // input string consisting of the UTF-8 password concatenated with the 8 bytes of user
        // validation salt. If the 32-byte result matches the first 32-bytes of the U string, this
        // is the user password.

        if self.compute_hash(password, user_validation_salt, None)? == hashed_user_password {
            // Compute an intermediate user key by computing a hash using algorithm 2.B with an
            // input string consisting of the UTF-8 owner password concatenated with the 8 bytes of
            // user key salt.
            let hash = self.compute_hash(password, user_key_salt, None)?;

            let mut key = [0u8; 32];
            key.copy_from_slice(&hash);

            // The 32-byte result is the key used to decrypt the 32-byte UE string using AES-256 in
            // CBC mode with no padding and an initialization vector of zero. The 32-byte result is
            // the file encryption key.
            let iv = [0u8; 16];
            let mut user_encrypted = self.user_encrypted.clone();
            let mut decryptor = Aes256CbcDec::new(&key.into(), &iv.into());

            for block in user_encrypted.as_chunks_mut::<16>().0 {
                decryptor.decrypt_block(aes_block_mut(block));
            }

            // Decrypt the 16-byte Perms string using AES-256 in EBC mode with an initialization
            // vector of zero and the file encryption key as the key. Verify that bytes 9-11 of the
            // result are the characters "a", "d", "b". Bytes 0-3 of the decrypted Perms entry,
            // treated as a little-endian integer, are the user permissions. They shall match the
            // value in the P key.
            //
            // i.e., use algorithm 13 to validate the permissions.
            self.validate_permissions(&user_encrypted)?;

            return Ok(user_encrypted);
        }

        Err(DecryptionError::IncorrectPassword)
    }

    /// Compute a hash (revision 6 and later).
    ///
    /// This implements Algorithm 2.B as described in ISO 32000-2:2020 (PDF 2.0).
    fn compute_hash<P, S>(&self, password: P, salt: S, user_key: Option<&[u8]>) -> Result<Vec<u8>, DecryptionError>
    where
        P: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let password = password.as_ref();
        let salt = salt.as_ref();

        // Take the SHA-256 hash of the original input to the algorithm and name the resulting 32
        // bytes, K.
        let mut hasher = Sha256::new();

        hasher.update(password);
        hasher.update(salt);

        if let Some(user_key) = user_key {
            hasher.update(user_key);
        }

        let mut k = hasher.finalize().to_vec();

        // Revision 5 uses a simplified hash algorithm that simply calculates the SHA-256 hash of
        // the original input to the algorithm.
        if self.revision == 5 {
            return Ok(k);
        }

        let mut k1 =
            Vec::with_capacity(64 * (password.len() + 64 + user_key.map(|user_key| user_key.len()).unwrap_or(0)));

        // Perform the following steps at least 64 times, until the value of the last byte in K is
        // less than or equal to (round number) - 32.
        for round in 1.. {
            // Make a new string K0 as follows:
            //
            // * When checking the owner password or creating the owner key, K0 is the concatenation of the input
            //   password, K, and the 48-byte user key.
            // * Otherwise, K0 is the concatenation of the input password and K.
            //
            // Next, set K1 to 64 repetitions of K0.
            k1.clear();

            for _ in 0..64 {
                k1.extend_from_slice(password);
                k1.extend_from_slice(&k);

                if let Some(user_key) = user_key {
                    k1.extend_from_slice(user_key);
                }
            }

            // Encrypt K1 with the AES-128 (CBC, no padding) algorithm, using the first 16 bytes of
            // K as the key, and the second 16 bytes of K as the initialization vector. The result
            // of this encryption is E.
            //
            // The 64 repetitions of K0 ensure that K1 is a multiple of 64 bytes, thus a multiple
            // of 16 bytes, i.e., it does not require padding.
            let key: &[u8; 16] = k[..16].try_into().expect("hash key must be 16 bytes");
            let iv: &[u8; 16] = k[16..32].try_into().expect("hash IV must be 16 bytes");

            let mut encryptor = Aes128CbcEnc::new(key.into(), iv.into());

            for block in k1.as_chunks_mut::<16>().0 {
                encryptor.encrypt_block(aes_block_mut(block));
            }

            let e = k1;

            // Taking the first 16 bytes of E as an unsigned big-endian integer, compute the
            // remainder, modulo 3. If the result is 0, the next hash used is SHA-256. If the
            // result is 1, the next hash used is SHA-384. If the result is 2, the next hash used
            // is SHA-256.
            //
            // Using the hash algorithm determined in the previous step, take the hash of E. The
            // result is a new value of K, which will be 32, 48 or 64 bytes in length.
            k = match e[..16].iter().map(|v| *v as u32).sum::<u32>() % 3 {
                0 => Sha256::digest(&e).to_vec(),
                1 => Sha384::digest(&e).to_vec(),
                2 => Sha512::digest(&e).to_vec(),
                _ => unreachable!(),
            };

            // Look at the very last byte of E. If the value of that byte (taken as an unsigned
            // integer) is greater than the round number - 32, repeat the round again.
            //
            // Repeat rounds until the value of the last byte is less than or equal to (round
            // number) - 32.
            if round >= 64 && e.last().copied().unwrap_or(0) as u32 <= round - 32 {
                break;
            }

            // Move e into k1 for the next round (to reuse k1).
            k1 = e;
        }

        // The first 32 bytes of the final K are the output of the algorithm.
        k.truncate(32);

        Ok(k)
    }

    /// Compute the encryption dictionary's O-entry value (revision 4 and earlier).
    ///
    /// This implements Algorithm 3 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    pub(crate) fn compute_hashed_owner_password_r4<O, U>(
        &self, owner_password: Option<O>, user_password: U,
    ) -> Result<Vec<u8>, DecryptionError>
    where
        O: AsRef<[u8]>,
        U: AsRef<[u8]>,
    {
        let user_password = user_password.as_ref();

        // Pad or truncate the owner string. If there is no owner password, use the user password
        // instead.
        let password = owner_password
            .as_ref()
            .map(|password| password.as_ref())
            .unwrap_or(user_password);

        // Pad or truncate the resulting password string to exactly 32 bytes. If the password string is
        // more than 32 bytes long, use only its first 32 bytes; if it is less than 32 bytes long, pad
        // it by appending the required number of additional bytes from the beginning of the following
        // padding string (see `PAD_BYTES`).
        //
        // That is, if the password is n bytes long, append the first 32 - n bytes of the padding
        // string to the end of the password string. If the password string is empty (zero-length),
        // meaning there is no user password, substitute the entire padding string in its place.
        //
        // i.e., we will simply calculate `len = min(password length, 32)` and use the first len bytes
        // of password and the first len bytes of `PAD_BYTES`.
        let len = password.len().min(32);

        // Initialize the MD5 hash function and pass the result as input to this function.
        let mut hasher = Md5::new();

        hasher.update(&password[..len]);
        hasher.update(&PAD_BYTES[..32 - len]);

        let mut hash = hasher.finalize();

        // (Security handlers of revision 3 or greater) Do the following 50 times: take the output from
        // the previous MD5 hash and pass it as input into a new MD5 hash.
        if self.revision >= 3 {
            for _ in 0..50 {
                hash = Md5::digest(hash);
            }
        }

        // Create an RC4 file encryption key using the first n bytes of the output from the final MD5
        // hash, where n shall always be 5 for security handlers of revision 2 but, for security
        // handlers of revision 3 or greater, shall depend on the value of the encryption dictionary's
        // Length entry.
        let n = if self.revision >= 3 {
            self.length.unwrap_or(40) / 8
        } else {
            5
        };

        // The maximum supported key length is 16 bytes (128 bits) due to the use of MD5.
        if n > 16 {
            return Err(DecryptionError::InvalidKeyLength);
        }

        // Pad or truncate the user password string to exactly 32 bytes. If the user password string is
        // more than 32 bytes long, use only its first 32 bytes; if it is less than 32 bytes long, pad
        // it by appending the required number of additional bytes from the beginning of the following
        // padding string (see `PAD_BYTES`).
        //
        // That is, if the password is n bytes long, append the first 32 - n bytes of the padding
        // string to the end of the password string. If the password string is empty (zero-length),
        // meaning there is no user password, substitute the entire padding string in its place.
        //
        // i.e., we will simply calculate `len = min(password length, 32)` and use the first len bytes
        // of password and the first len bytes of `PAD_BYTES`.
        let len = user_password.len().min(32);

        // Encrypt the result of the previous step using an RC4 encryption function with the RC4 file
        // encryption key obtained in the step before the previous step.
        let mut bytes = [0u8; 32];

        bytes[..len].copy_from_slice(&user_password[..len]);
        bytes[len..].copy_from_slice(&PAD_BYTES[..32 - len]);

        let mut result = Rc4::new(&hash[..n]).encrypt(bytes);

        // (Security handlers of revision 3 or greater) Do the following 19 times: Take the output from
        // the previous invocation of the RC4 function and pass it as input to a new invocation of the
        // function; use a file encryption key generated by taking each byte of the RC4 file encryption
        // key and performing an XOR (exclusive or) operation between that byte and the single-byte
        // value of the iteration counter (from 1 to 19).
        if self.revision >= 3 {
            let mut key = vec![0u8; n];

            for i in 1..=19 {
                for (in_byte, out_byte) in hash[..n].iter().zip(key.iter_mut()) {
                    *out_byte = in_byte ^ i;
                }

                result = Rc4::new(&key).encrypt(&result);
            }
        }

        // Store the output from the final invocation of the RC4 function as the value of the O entry
        // in the encryption dictionary.
        Ok(result)
    }

    /// Compute the encryption dictionary's U-entry value (revision 2).
    ///
    /// This implements Algorithm 4 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    pub(crate) fn compute_hashed_user_password_r2<U>(
        &self, doc: &Document, user_password: U,
    ) -> Result<Vec<u8>, DecryptionError>
    where
        U: AsRef<[u8]>,
    {
        // Create a file encryption key based on the user password string.
        let file_encryption_key = self.compute_file_encryption_key_r4(doc, user_password)?;

        // Encrypt the 32-byte padding string using an RC4 encryption function with the file encryption
        // key from the preceding step.
        let result = Rc4::new(&file_encryption_key).encrypt(PAD_BYTES);

        // Store the result of the previous step as the value of the U entry in the encryption dictionary.
        Ok(result)
    }

    /// Compute the encryption dictionary's U-entry value (revision 3 or 4).
    ///
    /// This implements Algorithm 5 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    pub(crate) fn compute_hashed_user_password_r3_r4<U>(
        &self, doc: &Document, user_password: U,
    ) -> Result<Vec<u8>, DecryptionError>
    where
        U: AsRef<[u8]>,
    {
        // Create a file encryption key based on the user password string.
        let file_encryption_key = self.compute_file_encryption_key_r4(doc, user_password)?;

        // Initialize the MD5 hash function and pass the 32-byte padding string.
        let mut hasher = Md5::new();

        hasher.update(PAD_BYTES);

        // Pass the first element of the file's file identifier array (the value of the ID entry in the
        // document's trailer dictionary) to the hash function and finish the hash.
        let file_id_0 = doc
            .trailer
            .get(b"ID")
            .map_err(|_| DecryptionError::MissingFileID)?
            .as_array()
            .map_err(|_| DecryptionError::InvalidType)?
            .first()
            .ok_or(DecryptionError::InvalidType)?
            .as_str()
            .map_err(|_| DecryptionError::InvalidType)?;
        hasher.update(file_id_0);

        let hash = hasher.finalize();

        // Encrypt the 16-byte result of the hash, using an RC4 encryption function with the file
        // encryption key.
        let mut result = Rc4::new(&file_encryption_key).encrypt(hash);

        // Do the following 19 times: Take the output from the previous invocation of the RC4 function
        // and pass it as input to a new invocation of the function; use a file encryption key
        // generated by taking each byte of the RC4 file encryption key and performing an XOR
        // (exclusive or) operation between that byte and the single-byte value of the iteration
        // counter (from 1 to 19).
        let mut key = vec![0u8; file_encryption_key.len()];

        for i in 1..=19 {
            for (in_byte, out_byte) in file_encryption_key.iter().zip(key.iter_mut()) {
                *out_byte = in_byte ^ i;
            }

            result = Rc4::new(&key).encrypt(&result);
        }

        // Append 16 bytes of arbitrary padding to the output from the final invocation of the RC4
        // function and store the 32-byte result as the value of the U entry in the encryption
        // dictionary.
        result.resize(32, 0);

        let mut rng = rand::rng();
        rng.fill(&mut result[16..]);

        Ok(result)
    }

    /// Authenticate the user password (revision 4 and earlier).
    ///
    /// This implements Algorithm 6 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    fn authenticate_user_password_r4<U>(&self, doc: &Document, user_password: U) -> Result<(), DecryptionError>
    where
        U: AsRef<[u8]>,
    {
        // Perform all but the last step of Algorithm 4 (security handlers of revision 2) or Algorithm
        // 5 (security handlers of revision 3 or 4) using the supplied password string to compute the
        // encryption dictionary's U-entry value.
        let hashed_user_password = match self.revision {
            2 => self.compute_hashed_user_password_r2(doc, &user_password)?,
            3 | 4 => self.compute_hashed_user_password_r3_r4(doc, &user_password)?,
            _ => return Err(DecryptionError::InvalidRevision),
        };

        // If the result of the previous step is equal to the value of the encryption dictionary's U
        // entry (comparing on the first 16 bytes in the case of security handlers of revision 3 or
        // greater), the password supplied is the correct user password.
        let len = match self.revision {
            3 | 4 => 16,
            _ => hashed_user_password.len(),
        };

        if self.user_value.len() < len {
            return Err(DecryptionError::InvalidHashLength);
        }

        if hashed_user_password[..len] != self.user_value[..len] {
            return Err(DecryptionError::IncorrectPassword);
        }

        Ok(())
    }

    /// Authenticate the owner password (revision 4 and earlier).
    ///
    /// This implements Algorithm 7 as described in ISO 32000-2:2020 (PDF 2.0).
    ///
    /// This algorithm is deprecated in PDF 2.0.
    fn authenticate_owner_password_r4<O>(&self, doc: &Document, owner_password: O) -> Result<(), DecryptionError>
    where
        O: AsRef<[u8]>,
    {
        // Pad or truncate the owner string. If there is no owner password, use the user password
        // instead.
        let password = owner_password.as_ref();

        // Pad or truncate the resulting password string to exactly 32 bytes. If the password string is
        // more than 32 bytes long, use only its first 32 bytes; if it is less than 32 bytes long, pad
        // it by appending the required number of additional bytes from the beginning of the following
        // padding string (see `PAD_BYTES`).
        //
        // That is, if the password is n bytes long, append the first 32 - n bytes of the padding
        // string to the end of the password string. If the password string is empty (zero-length),
        // meaning there is no user password, substitute the entire padding string in its place.
        //
        // i.e., we will simply calculate `len = min(password length, 32)` and use the first len bytes
        // of password and the first len bytes of `PAD_BYTES`.
        let len = password.len().min(32);

        // Initialize the MD5 hash function and pass the result as input to this function.
        let mut hasher = Md5::new();

        hasher.update(&password[..len]);
        hasher.update(&PAD_BYTES[..32 - len]);

        let mut hash = hasher.finalize();

        // (Security handlers of revision 3 or greater) Do the following 50 times: take the output from
        // the previous MD5 hash and pass it as input into a new MD5 hash.
        if self.revision >= 3 {
            for _ in 0..50 {
                hash = Md5::digest(hash);
            }
        }

        // Create an RC4 file encryption key using the first n bytes of the output from the final MD5
        // hash, where n shall always be 5 for security handlers of revision 2 but, for security
        // handlers of revision 3 or greater, shall depend on the value of the encryption dictionary's
        // Length entry.
        let n = if self.revision >= 3 {
            self.length.unwrap_or(40) / 8
        } else {
            5
        };

        // The maximum supported key length is 16 bytes (128 bits) due to the use of MD5.
        if n > 16 {
            return Err(DecryptionError::InvalidKeyLength);
        }

        // Decrypt the value of the encryption dictionary's O entry, using an RC4 encryption function
        // with the file encryption key to retrieve the user password.
        let mut result = self.owner_value.to_vec();

        // (Security handlers of revision 3 or greater) Do the following 19 times: Take the output from
        // the previous invocation of the RC4 function and pass it as input to a new invocation of the
        // function; use a file encryption key generated by taking each byte of the RC4 file encryption
        // key and performing an XOR (exclusive or) operation between that byte and the single-byte
        // value of the iteration counter (from 19 to 1).
        if self.revision >= 3 {
            let mut key = vec![0u8; n];

            for i in (1..=19).rev() {
                for (in_byte, out_byte) in hash[..n].iter().zip(key.iter_mut()) {
                    *out_byte = in_byte ^ i;
                }

                result = Rc4::new(&key).decrypt(&result);
            }
        }

        // (Security handler of revision 2 and the final step for revision 3 or greater) Decrypt the
        // value of the encryption dictionary's O entry, using an RC4 encryption function with the file
        // encryption key.
        result = Rc4::new(&hash[..n]).decrypt(&result);

        // The result of the previous step purports to be the user password. Authenticate this user
        // password using Algorithm 5. If it is correct, the password supplied is the correct owner
        // password.
        self.authenticate_user_password_r4(doc, &result)
    }

    /// Compute the encryption dictionary's U-entry value (revision 6).
    ///
    /// This implements Algorithm 8 as described in ISO 32000-2:2020 (PDF 2.0).
    pub(crate) fn compute_hashed_user_password_r6<K, U>(
        &self, file_encryption_key: K, user_password: U,
    ) -> Result<(Vec<u8>, Vec<u8>), DecryptionError>
    where
        K: AsRef<[u8]>,
        U: AsRef<[u8]>,
    {
        let file_encryption_key = file_encryption_key.as_ref();
        let user_password = user_password.as_ref();

        // Generate 16 random bytes of data using a strong random number generator. The first 8
        // bytes are the user validation salt. The second 8 bytes are the user key salt. Compute
        // the 32-byte hash using algorithm 2.B with an input string consisting of the UTF-8
        // password concatenated with the user validation salt. The 48-byte string consisting of
        // the 32-byte hash followed by the user validation salt followed by the user key salt is
        // stored as the U key.
        let mut user_value = [0u8; 48];
        let mut rng = rand::rng();

        rng.fill(&mut user_value[32..]);

        let user_validation_salt = &user_value[32..][..8];

        let mut input = Vec::with_capacity(user_password.len() + user_validation_salt.len());

        input.extend_from_slice(user_password);
        input.extend_from_slice(user_validation_salt);

        let hashed_user_password = self.compute_hash(user_password, user_validation_salt, None)?;
        user_value[..32].copy_from_slice(&hashed_user_password);

        // Compute the 32-byte hash using algorithm 2.B with an input string consisting of the
        // UTF-8 password concatenated with the user key salt.
        let user_key_salt = &user_value[40..][..8];

        input.clear();

        input.extend_from_slice(user_password);
        input.extend_from_slice(user_key_salt);

        let hash = self.compute_hash(user_password, user_key_salt, None)?;

        // Using this hash as the key, encrypt the file encryption key using AES-256 in CBC mode
        // with no padding and initialization vector of zero. The resulting 32-byte string is
        // stored as the UE key.
        let mut key = [0u8; 32];
        key.copy_from_slice(&hash);

        let iv = [0u8; 16];

        let mut user_encrypted = file_encryption_key.to_vec();
        let mut encryptor = Aes256CbcEnc::new(&key.into(), &iv.into());

        for block in user_encrypted.as_chunks_mut::<16>().0 {
            encryptor.encrypt_block(aes_block_mut(block));
        }

        Ok((user_value.to_vec(), user_encrypted))
    }

    /// Compute the encryption dictionary's O-entry value (revision 6).
    ///
    /// This implements Algorithm 9 as described in ISO 32000-2:2020 (PDF 2.0).
    pub(crate) fn compute_hashed_owner_password_r6<K, O>(
        &self, file_encryption_key: K, owner_password: O,
    ) -> Result<(Vec<u8>, Vec<u8>), DecryptionError>
    where
        K: AsRef<[u8]>,
        O: AsRef<[u8]>,
    {
        let file_encryption_key = file_encryption_key.as_ref();
        let owner_password = owner_password.as_ref();

        // Generate 16 random bytes of data using a strong random number generator. The first 8
        // bytes are the owner validation salt. The second 8 bytes are the owner key salt. Compute
        // the 32-byte hash using algorithm 2.B with an input string consisting of the UTF-8
        // password concatenated with the owner validation salt and then concatenated with the
        // 48-byte U string as generated in Algorithm 8. The 48-byte string consisting of the
        // 32-byte hash followed by the owner validation salt followed by the owner key salt is
        // stored as the O key.
        let mut owner_value = [0u8; 48];
        let mut rng = rand::rng();

        rng.fill(&mut owner_value[32..]);

        let owner_validation_salt = &owner_value[32..][..8];

        let hashed_owner_password = self.compute_hash(owner_password, owner_validation_salt, Some(&self.user_value))?;
        owner_value[..32].copy_from_slice(&hashed_owner_password);

        // Compute the 32-byte hash using algorithm 2.B with an input string consisting of the
        // UTF-8 password concatenated with the owner key salt.
        let owner_key_salt = &owner_value[40..][..8];

        let hash = self.compute_hash(owner_password, owner_key_salt, Some(&self.user_value))?;

        // Using this hash as the key, encrypt the file encryption key using AES-256 in CBC mode
        // with no padding and initialization vector of zero. The resulting 32-byte string is
        // stored as the OE key.
        let mut key = [0u8; 32];
        key.copy_from_slice(&hash);

        let iv = [0u8; 16];

        let mut owner_encrypted = file_encryption_key.to_vec();
        let mut encryptor = Aes256CbcEnc::new(&key.into(), &iv.into());

        for block in owner_encrypted.as_chunks_mut::<16>().0 {
            encryptor.encrypt_block(aes_block_mut(block));
        }

        Ok((owner_value.to_vec(), owner_encrypted))
    }

    /// Compute the encryption dictionary's Perms (permissions) value (revision 6 and later).
    ///
    /// This implements Algorithm 10 as described in ISO 32000-2:2020 (PDF 2.0).
    pub(crate) fn compute_permissions<K>(&self, file_encryption_key: K) -> Result<Vec<u8>, DecryptionError>
    where
        K: AsRef<[u8]>,
    {
        let file_encryption_key = file_encryption_key.as_ref();
        let mut bytes = [0u8; 16];

        // Record the 8 bytes of permission in the bytes 0-7 of the block, low order byte first.
        bytes[..8].copy_from_slice(&u64::to_le_bytes(self.permissions.bits()));

        // Set byte 8 to ASCII character "T" or "F" according to the EncryptMetadata boolean.
        bytes[8] = if self.encrypt_metadata { b'T' } else { b'F' };

        // Set bytes 9-11 to the ASCII characters "a", "d", "b".
        bytes[9..][..3].copy_from_slice(b"adb");

        // Set bytes 12-15 to 4 bytes of random data, which will be ignored.
        let mut rng = rand::rng();
        rng.fill(&mut bytes[12..][..4]);

        // Encrypt the 16-byte block using AES-256 in ECB mode with an initialization vector of
        // zero, using the file encryption key as the key.
        let mut key = [0u8; 32];
        key.copy_from_slice(file_encryption_key);

        let mut encryptor = Aes256EbcEnc::new(&key.into());

        for block in bytes.as_chunks_mut::<16>().0 {
            encryptor.encrypt_block(aes_block_mut(block));
        }

        // The result (16 bytes) is stored as the Perms string, and checked for validity when the
        // file is opened.
        Ok(bytes.to_vec())
    }

    /// Authenticate the user password (revision 6 and later).
    ///
    /// This implements Algorithm 11 as described in ISO 32000-2:2020 (PDF 2.0).
    fn authenticate_user_password_r6<U>(&self, user_password: U) -> Result<(), DecryptionError>
    where
        U: AsRef<[u8]>,
    {
        let mut user_password = user_password.as_ref();

        let hashed_user_password = &self.user_value[0..][..32];
        let user_validation_salt = &self.user_value[32..][..8];

        // Truncate the UTF-8 representation to 127 bytes if it is longer than 127 bytes.
        if user_password.len() > 127 {
            user_password = &user_password[..127];
        }

        // Test the password against the user key by computing a hash using algorithm 2.B with an
        // input string consisting of the UTF-8 password concatenated with the 8 bytes of user
        // validation salt. If the 32-byte result matches the first 32-bytes of the U string, this
        // is the user password.
        let mut input = Vec::with_capacity(user_password.len() + user_validation_salt.len());

        input.extend_from_slice(user_password);
        input.extend_from_slice(user_validation_salt);

        if self.compute_hash(user_password, user_validation_salt, None)? != hashed_user_password {
            return Err(DecryptionError::IncorrectPassword);
        }

        Ok(())
    }

    /// Authenticate the owner password (revision 6 and later).
    ///
    /// This implements Algorithm 12 as described in ISO 32000-2:2020 (PDF 2.0).
    fn authenticate_owner_password_r6<O>(&self, owner_password: O) -> Result<(), DecryptionError>
    where
        O: AsRef<[u8]>,
    {
        let mut owner_password = owner_password.as_ref();

        let hashed_owner_password = &self.owner_value[0..][..32];
        let owner_validation_salt = &self.owner_value[32..][..8];

        // Truncate the UTF-8 representation to 127 bytes if it is longer than 127 bytes.
        if owner_password.len() > 127 {
            owner_password = &owner_password[..127];
        }

        // Test the password against the owner key by computing a hash using algorithm 2.B with an
        // input string consisting of the UTF-8 password concatenated with the 8 bytes of owner
        // validation salt and the 48 byte U string. If the 32-byte result matches the first
        // 32-bytes of the O string, this is the owner password.
        let mut input = Vec::with_capacity(owner_password.len() + owner_validation_salt.len());

        input.extend_from_slice(owner_password);
        input.extend_from_slice(owner_validation_salt);

        if self.compute_hash(owner_password, owner_validation_salt, Some(&self.user_value))? != hashed_owner_password {
            return Err(DecryptionError::IncorrectPassword);
        }

        Ok(())
    }

    /// Validate the permissions (revision 6 and later).
    ///
    /// This implements Algorithm 13 as described in ISO 32000-2:2020 (PDF 2.0).
    fn validate_permissions<K>(&self, file_encryption_key: K) -> Result<(), DecryptionError>
    where
        K: AsRef<[u8]>,
    {
        let file_encryption_key = file_encryption_key.as_ref();

        // Decrypt the 16 byte Perms string using AES-256 in ECB mode with an initialization vector
        // of zero and the file encryption key as the key.
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&self.permission_encrypted);

        let mut key = [0u8; 32];
        key.copy_from_slice(file_encryption_key);

        let mut decryptor = Aes256EbcDec::new(&key.into());

        for block in bytes.as_chunks_mut::<16>().0 {
            decryptor.decrypt_block(aes_block_mut(block));
        }

        // Verify that bytes 9-11 of the result are the characters "a", "d", "b".
        if &bytes[9..][..3] != b"adb" {
            return Err(DecryptionError::IncorrectPassword);
        }

        // Bytes 0-3 of the decrypted Perms entry, treated as a little-endian integer, are the
        // user permissions. They should match the value in the P key.
        if bytes[..3] != u64::to_le_bytes(self.permissions.bits())[..3] {
            return Err(DecryptionError::IncorrectPassword);
        }

        // Byte 8 should match the ASCII character "T" or "F" according to the boolean value of the
        // EncryptMetadata key.
        if bytes[8] != if self.encrypt_metadata { b'T' } else { b'F' } {
            return Err(DecryptionError::IncorrectPassword);
        }

        Ok(())
    }

    /// Sanitize the password.
    pub fn sanitize_password(&self, password: &str) -> Result<Vec<u8>, DecryptionError> {
        match self.revision {
            2..=4 => self.sanitize_password_r4(password),
            5..=6 => self.sanitize_password_r6(password),
            _ => Err(DecryptionError::UnsupportedRevision),
        }
    }

    /// Compute the file encryption key used to encrypt/decrypt the document.
    pub fn compute_file_encryption_key<P>(&self, doc: &Document, password: P) -> Result<Vec<u8>, DecryptionError>
    where
        P: AsRef<[u8]>,
    {
        match self.revision {
            2..=4 => self.compute_file_encryption_key_r4(doc, password),
            5..=6 => self.compute_file_encryption_key_r6(password),
            _ => Err(DecryptionError::UnsupportedRevision),
        }
    }

    /// Authenticate the owner password.
    pub fn authenticate_user_password<U>(&self, doc: &Document, user_password: U) -> Result<(), DecryptionError>
    where
        U: AsRef<[u8]>,
    {
        match self.revision {
            2..=4 => self.authenticate_user_password_r4(doc, user_password),
            5..=6 => self.authenticate_user_password_r6(user_password),
            _ => Err(DecryptionError::UnsupportedRevision),
        }
    }

    /// Authenticate the owner password.
    pub fn authenticate_owner_password<O>(&self, doc: &Document, owner_password: O) -> Result<(), DecryptionError>
    where
        O: AsRef<[u8]>,
    {
        match self.revision {
            2..=4 => self.authenticate_owner_password_r4(doc, owner_password),
            5..=6 => self.authenticate_owner_password_r6(owner_password),
            _ => Err(DecryptionError::UnsupportedRevision),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Permissions;
    use crate::creator::tests::create_document;
    use crate::encryption::PasswordAlgorithm;
    use rand::RngExt as _;

    #[test]
    fn authenticate_password_r2() {
        let document = create_document();

        let mut algorithm = PasswordAlgorithm {
            encrypt_metadata: true,
            length: None,
            version: 1,
            revision: 2,
            permissions: Permissions::all(),
            ..Default::default()
        };

        let owner_password = "owner";
        let user_password = "user";

        // Sanitize the passwords.
        let owner_password = algorithm.sanitize_password_r4(owner_password).unwrap();
        let user_password = algorithm.sanitize_password_r4(user_password).unwrap();

        // Compute the hashed values.
        algorithm.owner_value = algorithm
            .compute_hashed_owner_password_r4(Some(&owner_password), &user_password)
            .unwrap();

        algorithm.user_value = algorithm
            .compute_hashed_user_password_r2(&document, &user_password)
            .unwrap();

        // Assert that the correct passwords authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, &owner_password)
                .is_ok()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, &user_password)
                .is_ok()
        );

        // Assert that the swapped passwords do not authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, user_password)
                .is_err()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, owner_password)
                .is_err()
        );
    }

    #[test]
    fn authenticate_password_r3() {
        let document = create_document();

        let mut algorithm = PasswordAlgorithm {
            encrypt_metadata: true,
            length: Some(40),
            version: 2,
            revision: 3,
            permissions: Permissions::all(),
            ..Default::default()
        };

        let owner_password = "owner";
        let user_password = "user";

        // Sanitize the passwords.
        let owner_password = algorithm.sanitize_password_r4(owner_password).unwrap();
        let user_password = algorithm.sanitize_password_r4(user_password).unwrap();

        // Compute the hashed values.
        algorithm.owner_value = algorithm
            .compute_hashed_owner_password_r4(Some(&owner_password), &user_password)
            .unwrap();

        algorithm.user_value = algorithm
            .compute_hashed_user_password_r3_r4(&document, &user_password)
            .unwrap();

        // Assert that the correct passwords authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, &owner_password)
                .is_ok()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, &user_password)
                .is_ok()
        );

        // Assert that the swapped passwords do not authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, user_password)
                .is_err()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, owner_password)
                .is_err()
        );
    }

    #[test]
    fn authenticate_password_r4() {
        let document = create_document();

        let mut algorithm = PasswordAlgorithm {
            encrypt_metadata: true,
            length: Some(128),
            version: 4,
            revision: 4,
            permissions: Permissions::all(),
            ..Default::default()
        };

        let owner_password = "owner";
        let user_password = "user";

        // Sanitize the passwords.
        let owner_password = algorithm.sanitize_password_r4(owner_password).unwrap();
        let user_password = algorithm.sanitize_password_r4(user_password).unwrap();

        // Compute the hashed values.
        algorithm.owner_value = algorithm
            .compute_hashed_owner_password_r4(Some(&owner_password), &user_password)
            .unwrap();

        algorithm.user_value = algorithm
            .compute_hashed_user_password_r3_r4(&document, &user_password)
            .unwrap();

        // Assert that the correct passwords authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, &owner_password)
                .is_ok()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, &user_password)
                .is_ok()
        );

        // Assert that the swapped passwords do not authenticate.
        assert!(
            algorithm
                .authenticate_owner_password_r4(&document, user_password)
                .is_err()
        );
        assert!(
            algorithm
                .authenticate_user_password_r4(&document, owner_password)
                .is_err()
        );
    }

    #[test]
    fn authenticate_password_r5() {
        let mut algorithm = PasswordAlgorithm {
            encrypt_metadata: true,
            version: 5,
            revision: 5,
            permissions: Permissions::all(),
            ..Default::default()
        };

        let owner_password = "owner";
        let user_password = "user";

        // Sanitize the passwords.
        let owner_password = algorithm.sanitize_password_r6(owner_password).unwrap();
        let user_password = algorithm.sanitize_password_r6(user_password).unwrap();

        // Compute the hashed values.
        let mut file_encryption_key = [0u8; 32];

        let mut rng = rand::rng();
        rng.fill(&mut file_encryption_key);

        let (user_value, user_encrypted) = algorithm
            .compute_hashed_user_password_r6(file_encryption_key, &user_password)
            .unwrap();

        algorithm.user_value = user_value;
        algorithm.user_encrypted = user_encrypted;

        let (owner_value, owner_encrypted) = algorithm
            .compute_hashed_owner_password_r6(file_encryption_key, &owner_password)
            .unwrap();

        algorithm.owner_value = owner_value;
        algorithm.owner_encrypted = owner_encrypted;

        algorithm.permission_encrypted = algorithm.compute_permissions(file_encryption_key).unwrap();

        // Assert that the correct passwords authenticate.
        assert!(algorithm.authenticate_owner_password_r6(&owner_password).is_ok());
        assert!(algorithm.authenticate_user_password_r6(&user_password).is_ok());

        // Assert that the swapped passwords do not authenticate.
        assert!(algorithm.authenticate_owner_password_r6(&user_password).is_err());
        assert!(algorithm.authenticate_user_password_r6(&owner_password).is_err());

        // Assert that the permissions validate correctly.
        assert!(algorithm.validate_permissions(file_encryption_key).is_ok());

        // Assert that the file encryption key is equal for the owner password.
        let key = algorithm.compute_file_encryption_key_r6(&owner_password).unwrap();
        assert_eq!(&file_encryption_key[..], key);

        // Assert that the file encryption key is equal for the user password.
        let key = algorithm.compute_file_encryption_key_r6(&user_password).unwrap();
        assert_eq!(&file_encryption_key[..], key);
    }

    #[test]
    fn authenticate_password_r6() {
        let mut algorithm = PasswordAlgorithm {
            encrypt_metadata: true,
            version: 5,
            revision: 6,
            permissions: Permissions::all(),
            ..Default::default()
        };

        let owner_password = "owner";
        let user_password = "user";

        // Sanitize the passwords.
        let owner_password = algorithm.sanitize_password_r6(owner_password).unwrap();
        let user_password = algorithm.sanitize_password_r6(user_password).unwrap();

        // Compute the hashed values.
        let mut file_encryption_key = [0u8; 32];

        let mut rng = rand::rng();
        rng.fill(&mut file_encryption_key);

        let (user_value, user_encrypted) = algorithm
            .compute_hashed_user_password_r6(file_encryption_key, &user_password)
            .unwrap();

        algorithm.user_value = user_value;
        algorithm.user_encrypted = user_encrypted;

        let (owner_value, owner_encrypted) = algorithm
            .compute_hashed_owner_password_r6(file_encryption_key, &owner_password)
            .unwrap();

        algorithm.owner_value = owner_value;
        algorithm.owner_encrypted = owner_encrypted;

        algorithm.permission_encrypted = algorithm.compute_permissions(file_encryption_key).unwrap();

        // Assert that the correct passwords authenticate.
        assert!(algorithm.authenticate_owner_password_r6(&owner_password).is_ok());
        assert!(algorithm.authenticate_user_password_r6(&user_password).is_ok());

        // Assert that the swapped passwords do not authenticate.
        assert!(algorithm.authenticate_owner_password_r6(&user_password).is_err());
        assert!(algorithm.authenticate_user_password_r6(&owner_password).is_err());

        // Assert that the permissions validate correctly.
        assert!(algorithm.validate_permissions(file_encryption_key).is_ok());

        // Assert that the file encryption key is equal for the owner password.
        let key = algorithm.compute_file_encryption_key_r6(&owner_password).unwrap();
        assert_eq!(&file_encryption_key[..], key);

        // Assert that the file encryption key is equal for the user password.
        let key = algorithm.compute_file_encryption_key_r6(&user_password).unwrap();
        assert_eq!(&file_encryption_key[..], key);
    }

    /// Some PDF writers (e.g. Adobe) pad /O and /U to 127 bytes with trailing
    /// zeros instead of the spec-required 48. Verify that `try_from` accepts
    /// these and truncates to the correct 48-byte length.
    #[test]
    fn r6_padded_owner_user_values_accepted() {
        use crate::{Document, Object, StringFormat, dictionary};

        let mut doc = Document::with_version("2.0");

        // Build valid 48-byte /O and /U (contents don't matter for parsing).
        let o_48 = vec![0xAAu8; 48];
        let u_48 = vec![0xBBu8; 48];

        // Pad to 127 bytes with trailing zeros — mimics real-world Adobe PDFs.
        let mut o_127 = o_48.clone();
        o_127.resize(127, 0u8);
        let mut u_127 = u_48.clone();
        u_127.resize(127, 0u8);

        let encrypt_dict = dictionary! {
            "Filter" => "Standard",
            "V" => Object::Integer(5),
            "R" => Object::Integer(6),
            "Length" => Object::Integer(256),
            "O" => Object::String(o_127, StringFormat::Literal),
            "OE" => Object::String(vec![0xCCu8; 32], StringFormat::Literal),
            "U" => Object::String(u_127, StringFormat::Literal),
            "UE" => Object::String(vec![0xDDu8; 32], StringFormat::Literal),
            "P" => Object::Integer(-3388),
            "Perms" => Object::String(vec![0xEEu8; 16], StringFormat::Literal)
        };

        let encrypt_id = doc.add_object(encrypt_dict);
        doc.trailer.set("Encrypt", Object::Reference(encrypt_id));

        let algo = PasswordAlgorithm::try_from(&doc).expect("should accept padded /O and /U longer than 48 bytes");

        // Verify the values were truncated to the spec-required 48 bytes.
        assert_eq!(algo.owner_value.len(), 48);
        assert_eq!(algo.user_value.len(), 48);
        assert_eq!(&algo.owner_value, &o_48);
        assert_eq!(&algo.user_value, &u_48);
    }

    /// Verify that /O and /U values shorter than 48 bytes are still rejected
    /// for R >= 5.
    #[test]
    fn r6_short_owner_user_values_rejected() {
        use crate::{Document, Object, StringFormat, dictionary};

        let mut doc = Document::with_version("2.0");

        let encrypt_dict = dictionary! {
            "Filter" => "Standard",
            "V" => Object::Integer(5),
            "R" => Object::Integer(6),
            "Length" => Object::Integer(256),
            "O" => Object::String(vec![0xAAu8; 47], StringFormat::Literal),
            "OE" => Object::String(vec![0xCCu8; 32], StringFormat::Literal),
            "U" => Object::String(vec![0xBBu8; 48], StringFormat::Literal),
            "UE" => Object::String(vec![0xDDu8; 32], StringFormat::Literal),
            "P" => Object::Integer(-3388),
            "Perms" => Object::String(vec![0xEEu8; 16], StringFormat::Literal)
        };

        let encrypt_id = doc.add_object(encrypt_dict);
        doc.trailer.set("Encrypt", Object::Reference(encrypt_id));

        assert!(
            PasswordAlgorithm::try_from(&doc).is_err(),
            "should reject /O shorter than 48 bytes"
        );
    }
}
